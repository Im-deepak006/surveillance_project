"""Create a saved-artifact-only visual review of the Moment-DETR pilot.

No model is loaded or invoked. Full-video person boxes/IDs are read from the
saved ByteTrack result, candidate intervals/scores from the saved A/D and T
results, and crop/text similarities from the existing crop diagnostic cache.
"""
import argparse
import html
import json
import math
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "visual_review_20261023"
DEFAULTS = {
    "video": Path(r"D:\yolo_enhance\video\MOT17-04-SDP-raw.webm"),
    "annotations": ROOT / "evaluation_data" / "MOT17_04_SINGLE_VIDEO_GROUND_TRUTH.json",
    "proposals": ROOT / "outputs" / "proposal_diagnosis_mot17_04_verified_20261023" / "results.json",
    "tracks": ROOT / "outputs" / "tracking_consensus_bytetrack_20261023_2310" / "results.json",
    "crop_diagnostics": ROOT / "outputs" / "automated_visual_diagnostics_mot17_04_verified_20261023" / "results.json",
}


def _fmt(value, digits=3):
    if value is None:
        return "missing"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{value:.{digits}f}" if math.isfinite(value) else "missing"


def format_score_track_label(score_name, score, track_id=None, identity_verified=False):
    """Label score provenance without implying predicted IDs are true identity."""
    label = f"{score_name}: {_fmt(score)}"
    if track_id is not None:
        identity = "verified identity" if identity_verified else "predicted track ID"
        label += f" · {identity} {track_id}"
    elif score is None:
        label += " · unavailable in saved artifacts"
    return label


def render_interval_timeline(gt_intervals, predictions, duration_seconds, width=900):
    """Return an inline SVG interval plot; prediction values map name to (s,e,IoU)."""
    duration = max(float(duration_seconds or 0), 0.01)
    height = 46 + 30 * (1 + len(predictions))
    left, right = 150, width - 18
    plot_w = right - left
    scale = lambda t: left + max(0.0, min(duration, float(t))) / duration * plot_w
    out = [f'<svg class="timeline" viewBox="0 0 {width} {height}" role="img" aria-label="Ground truth and proposal intervals">']
    out.append(f'<line x1="{left}" y1="24" x2="{right}" y2="24" stroke="#666"/>')
    for tick in range(int(duration // 5) + 1):
        t = tick * 5
        x = scale(t)
        out.append(f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{height-8}" stroke="#ddd"/>')
        out.append(f'<text x="{x:.1f}" y="15" text-anchor="middle">{t}s</text>')
    rows = [("Ground truth", gt_intervals, "#25a56a")]
    rows.extend((name, [value[:2]], color) for name, value, color in predictions)
    for index, (name, intervals, color) in enumerate(rows):
        y = 39 + index * 30
        out.append(f'<text x="{left-8}" y="{y+12}" text-anchor="end">{html.escape(name)}</text>')
        for start, end in intervals or []:
            if start is None or end is None:
                continue
            x1, x2 = scale(start), scale(end)
            out.append(f'<rect x="{x1:.1f}" y="{y}" width="{max(1,x2-x1):.1f}" height="16" rx="3" fill="{color}"/>')
    out.append("</svg>")
    return "".join(out)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw, ih = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def _load_json(path, missing):
    if not path.is_file():
        missing.append(f"Missing input: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        missing.append(f"Unreadable input {path}: {exc}")
        return None


def _index(rows):
    return {row["qid"]: row for row in rows or [] if isinstance(row, dict) and row.get("qid")}


def _track_frame_index(track_data):
    index = {}
    if not track_data:
        return index
    for track in track_data.get("tracker", {}).get("track_observations", []):
        tid = track.get("track_id")
        for obs in track.get("observations", []):
            index.setdefault(int(obs["frame_index"]), []).append({**obs, "track_id": tid})
    return index


def _track_match(frame_index, box, tracks_by_frame, threshold=.5):
    rows = tracks_by_frame.get(int(frame_index), [])
    if not rows:
        return None
    best = max(rows, key=lambda row: box_iou(box, row["box_xyxy"]))
    return best if box_iou(box, best["box_xyxy"]) >= threshold else None


def _write_crop_from_video(cap, timestamp, box, destination):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, float(timestamp)) * 1000)
    ok, frame = cap.read()
    if not ok:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(x))) for x in box]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(cv2.imwrite(str(destination), frame[y1:y2, x1:x2]))


def _draw_review_frame(cap, frame_index, fps, boxes, title, interval, iou, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    for row in boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in row["box_xyxy"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 220, 100), 2)
        cv2.putText(frame, f"pred ID {row['track_id']}", (x1, max(18, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 30, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, f"pred ID {row['track_id']}", (x1, max(18, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (100, 255, 140), 1, cv2.LINE_AA)
    stamp = int(frame_index) / fps if fps else 0
    cv2.rectangle(frame, (0, 0), (w, 76), (15, 20, 30), -1)
    title_lines = title.split("\n", 1)
    label = f"{title_lines[0]} | t={stamp:.2f}s | interval [{interval[0]:.2f},{interval[1]:.2f}] | IoU={_fmt(iou)}"
    cv2.putText(frame, label[:145], (12, 20), cv2.FONT_HERSHEY_SIMPLEX, .47, (255,255,255), 1, cv2.LINE_AA)
    query_line = title_lines[1] if len(title_lines) > 1 else ""
    cv2.putText(frame, query_line[:130], (12, 44), cv2.FONT_HERSHEY_SIMPLEX, .43, (220,230,245), 1, cv2.LINE_AA)
    cv2.putText(frame, "ByteTrack IDs are predicted IDs, not verified identities", (12, 67),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (120,220,255), 1, cv2.LINE_AA)
    if (w, h) != size:
        frame = cv2.resize(frame, size)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_','-')}", default=str(default))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-video-frames", type=int, default=15,
                        help="Maximum short highlight sequence length; one annotated representative frame per query/method")
    args = parser.parse_args()
    paths = {key: Path(getattr(args, key)) for key in DEFAULTS}
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review directory: {output}")
    output.mkdir(parents=True)
    assets = output / "assets"
    assets.mkdir()
    missing = []
    loaded = {key: _load_json(path, missing) for key, path in paths.items() if key != "video"}
    video_path = paths["video"]
    video_exists = video_path.is_file()
    if not video_exists:
        missing.append(f"Missing video: {video_path}; annotated video and extracted crops cannot be created")
    source_queries = _index((loaded.get("proposals") or {}).get("queries"))
    track_queries = _index((loaded.get("tracks") or {}).get("queries"))
    diag_queries = _index((loaded.get("crop_diagnostics") or {}).get("queries"))
    annotation_queries = _index((loaded.get("annotations") or {}).get("queries"))
    fps = 30.0
    if loaded.get("tracks"):
        fps = float(loaded["tracks"].get("tracker", {}).get("video_fps") or 30.0)
    tracks_by_frame = _track_frame_index(loaded.get("tracks"))
    cap = cv2.VideoCapture(str(video_path)) if video_exists else None
    video_ok = bool(cap is not None and cap.isOpened())
    if video_exists and not video_ok:
        missing.append(f"Video exists but OpenCV could not decode it: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) if video_ok else fps
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps) if video_ok and video_fps else 35.0
    review_rows, video_rows = [], []

    qids = list(annotation_queries or source_queries or track_queries)
    for qid in qids:
        ann = annotation_queries.get(qid, {})
        src = source_queries.get(qid, {})
        trq = track_queries.get(qid, {})
        dq = diag_queries.get(qid, {})
        text = ann.get("query", src.get("query", trq.get("query", "")))
        gt = ann.get("ground_truth_intervals_seconds", src.get("ground_truth_intervals_seconds", trq.get("ground_truth_intervals_seconds", [])))
        methods = {}
        source_conditions = src.get("conditions", {})
        for method, source_name in (("A", "A_temporal_only"), ("D", "D_person_aware_evidence_fusion")):
            row = source_conditions.get(source_name)
            if row:
                methods[method] = {**row, "start": row.get("start"), "end": row.get("end")}
        t_conditions = trq.get("conditions", {})
        if t_conditions.get("T"):
            methods["T"] = t_conditions["T"]
        prediction_tuples = [(m, (v.get("start"), v.get("end"), v.get("temporal_iou")),
                              {"A":"#5286df", "D":"#ed9f32", "T":"#ab71d0"}[m])
                             for m,v in methods.items() if v.get("start") is not None and v.get("end") is not None]
        tracks_method_candidates = {m: next((c for c in trq.get("all_candidate_scores", [])
            if abs(float(c.get("start", -1))-float(v.get("start", -2))) < 1e-4 and
               abs(float(c.get("end", -1))-float(v.get("end", -2))) < 1e-4), None)
            for m,v in methods.items()}
        crops = []
        diag = dq.get("saved_A_B_D_diagnostics", {})
        for method in ("A", "D"):
            drow = diag.get(method, {})
            if not drow or not drow.get("person_crop_scores"):
                missing.append(f"{qid}/{method}: saved crop-score diagnostics unavailable")
                continue
            phrase = drow.get("appearance_phrase")
            scores = drow["person_crop_scores"]
            candidates = []
            if phrase:
                candidates.append(("appearance", max(scores, key=lambda x: x.get("similarities", {}).get(phrase, -1))))
            selected_indices = set(drow.get("saved_scorer_selected_crop_indices_frame_local", []))
            for item in scores:
                if item.get("selected_by_saved_person_object_scorer") or item.get("crop_index_frame_local") in selected_indices:
                    candidates.append(("person-object", item))
            for role, crop in candidates:
                ci = crop.get("crop_index_frame_local")
                original = paths["crop_diagnostics"].parent / "crops" / f"{qid}_{method}_crop{ci}.jpg"
                copied = None
                if original.is_file():
                    dest = assets / f"{qid}_{method}_{role.replace('-','_')}_crop{ci}.jpg"
                    shutil.copy2(original, dest)
                    copied = dest.name
                else:
                    missing.append(f"Missing cached crop image: {original}")
                ts = float(drow.get("representative_timestamp_seconds", 0))
                fi = int(ts * fps)
                match = _track_match(fi, crop.get("box_xyxy", []), tracks_by_frame) if len(crop.get("box_xyxy", [])) == 4 else None
                crops.append({"method": method, "role": role, "path": copied,
                    "timestamp_seconds": ts, "frame_index": fi, "track_id": match.get("track_id") if match else None,
                    "track_id_label": "predicted track ID" if match else None,
                    "score": (crop.get("similarities") or {}).get(phrase) if role == "appearance" else
                             max((crop.get("similarities") or {}).values(), default=None),
                    "score_label": role, "crop_index_frame_local": ci,
                    "box_xyxy": crop.get("box_xyxy"),
                    "object_phrase_scores": crop.get("similarities") if role == "person-object" else None,
                    "source": "existing crop diagnostic; ID matched to saved ByteTrack detection by same-frame bbox IoU >= 0.5"})
        for method in ("A", "D", "T"):
            if method not in methods:
                missing.append(f"{qid}/{method}: saved result unavailable")
                continue
            if method != "T":
                continue
            candidate = tracks_method_candidates.get("T")
            evidence = ((candidate or {}).get("track_phrase_evidence") or methods[method].get("track_phrase_evidence") or {})
            if not evidence:
                missing.append(f"{qid}/T: saved track phrase evidence unavailable")
                continue
            appearances = evidence.get("appearance", {})
            actions = evidence.get("action", {})
            app_id, act_id = appearances.get("best_track_id"), actions.get("best_track_id")
            for role, track_id, evidence_row in (("appearance", app_id, appearances), ("action", act_id, actions)):
                track_item = evidence_row.get("tracks", {}).get(str(track_id), {})
                frame_ids = track_item.get("frame_indices", [])
                # Prefer a selected-evidence sample within GT; otherwise use latest saved sample.
                possible = [fi for fi in frame_ids if any(float(a) <= fi/fps <= float(b) for a,b in gt)]
                fi = int(possible[-1] if possible else (frame_ids[-1] if frame_ids else -1))
                obs = next((r for r in tracks_by_frame.get(fi, []) if r.get("track_id") == track_id), None)
                if not obs or not video_ok:
                    missing.append(f"{qid}/T/{role}: crop could not be extracted from saved track box")
                    continue
                name = f"{qid}_T_{role}_predtrack{track_id}_f{fi}.jpg"
                if not _write_crop_from_video(cap, fi / fps, obs["box_xyxy"], assets / name):
                    missing.append(f"{qid}/T/{role}: source frame/crop unavailable at {fi/fps:.2f}s")
                    continue
                crops.append({"method":"T", "role":role, "path":name, "timestamp_seconds":fi/fps,
                    "frame_index":fi, "track_id":track_id, "track_id_label":"predicted track ID",
                    "score":evidence_row.get("score"), "score_label":role,
                    "identity_verified":False, "box_xyxy":obs["box_xyxy"],
                    "source":"existing ByteTrack detection; crop only, no model inference"})
            interval = (methods[method].get("start"), methods[method].get("end"))
            video_rows.append({"qid":qid,"query":text,"method":"T","interval":interval,
                               "iou":methods[method].get("temporal_iou")})

        for method in ("A", "D"):
            if methods.get(method):
                video_rows.append({"qid":qid,"query":text,"method":method,
                    "interval":(methods[method].get("start"),methods[method].get("end")),
                    "iou":methods[method].get("temporal_iou")})
        # retain the natural A/D/T order in the short video
        video_rows[-3:] = sorted(video_rows[-3:], key=lambda x: (x["qid"], {"A":0,"D":1,"T":2}[x["method"]])) if len(video_rows)>=3 else video_rows
        review_rows.append({"qid":qid,"query":text,"ground_truth_intervals_seconds":gt,
            "target_person_identity_status":"unverified; no ground-truth person tracks",
            "methods":methods,"crops":crops,
            "appearance_action_track_match": next((
                {"appearance_track_id":v.get("appearance",{}).get("best_track_id"),
                 "action_track_id":v.get("action",{}).get("best_track_id"),
                 "different_tracks":v.get("appearance",{}).get("best_track_id") != v.get("action",{}).get("best_track_id"),
                 "label":"predicted track IDs; identity not verified"}
                for v in [(tracks_method_candidates.get("T") or {}).get("track_phrase_evidence", {})]
                if v), None)})

    video_out = output / "diagnostic_highlights.mp4"
    video_status = "not created"
    if video_ok and video_rows:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width,height))
        if writer.isOpened():
            written = 0
            for item in video_rows[:max(0,args.max_video_frames)]:
                interval = item["interval"]
                if interval[0] is None or interval[1] is None:
                    continue
                timestamp = (float(interval[0])+float(interval[1]))/2
                fi = int(round(timestamp*fps))
                frame = _draw_review_frame(cap, fi, fps, tracks_by_frame.get(fi, []),
                    f"{item['qid']} · {item['method']}\n{item['query']}", interval,item["iou"],(width,height))
                if frame is not None:
                    for _ in range(5):
                        writer.write(frame)
                    written += 1
            writer.release()
            if written:
                video_status = f"created; {written} representative annotated frames held for 1 second each"
            else:
                video_out.unlink(missing_ok=True)
                video_status = "not created; no source frames could be decoded at selected timestamps"
        else:
            video_status = "not created; OpenCV has no usable mp4v writer"
    else:
        missing.append("No diagnostic video: source video or saved query intervals unavailable")

    if cap:
        cap.release()
    (output / "review_data.json").write_text(json.dumps({"video":str(video_path),"fps":fps,
        "video_duration_seconds":duration,"track_ids_are_ground_truth":False,
        "missing_inputs_or_evidence":missing,"video_status":video_status,"queries":review_rows},indent=2),encoding="utf-8")
    report = ["<!doctype html><html><head><meta charset='utf-8'><title>Moment-DETR visual review</title>",
      "<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}section{border-top:1px solid #ccd3dc;padding:1rem 0}table{border-collapse:collapse;width:100%;margin:.8rem 0}td,th{border:1px solid #ccd3dc;padding:.45rem;text-align:left}th{background:#eef2f6}.warn{background:#fff4d5;padding:.7rem}.crops{display:flex;gap:1rem;flex-wrap:wrap}.crop{width:190px}.crop img{max-width:180px;max-height:260px;border:1px solid #aaa}.timeline{width:100%;font:12px system-ui}.scores{font-family:monospace}.flag{color:#9b2c2c;font-weight:bold}</style></head><body>",
      "<h1>Moment-DETR / YOLO11 / OpenCLIP / ByteTrack visual review</h1>",
      f"<p>Saved artifacts only. Diagnostic video: {html.escape(video_status)}. No model inference was run.</p>",
      "<p class='warn'><b>Identity caution:</b> boxes and IDs are predicted ByteTrack outputs, not verified ground-truth identities. This review does not assert that a box belongs to the queried person or that a person owns a nearby object.</p>"]
    if missing:
        report.append("<details><summary>Missing inputs or evidence (not fabricated)</summary><ul>"+"".join(f"<li>{html.escape(x)}</li>" for x in missing)+"</ul></details>")
    if video_status.startswith("created"):
        report.append("<video controls preload='metadata' width='960'><source src='diagnostic_highlights.mp4' type='video/mp4'></video>")
    report.append("<p><a href='review_data.json'>Machine-readable review data</a></p>")
    for row in review_rows:
        preds=[]
        for m,color in (("A","#5286df"),("D","#ed9f32"),("T","#ab71d0")):
            v=row["methods"].get(m)
            if v and v.get("start") is not None and v.get("end") is not None:
                preds.append((m,(v["start"],v["end"],v.get("temporal_iou")),color))
        report.append(f"<section id='{html.escape(row['qid'])}'><h2>{html.escape(row['qid'])}</h2><p>{html.escape(row['query'])}</p>")
        report.append(render_interval_timeline(row["ground_truth_intervals_seconds"],preds,duration))
        report.append("<table><tr><th>Method</th><th>Selected interval (s)</th><th>Temporal IoU</th><th>Available scores</th><th>Ranking inputs / track evidence</th></tr>")
        for m in ("A","D","T"):
            v=row["methods"].get(m)
            if not v:
                report.append(f"<tr><td>{m}</td><td colspan='4'>Missing saved result</td></tr>"); continue
            interval=f"[{_fmt(v.get('start'),2)}, {_fmt(v.get('end'),2)}]"
            scores="; ".join(f"{key}={_fmt(v.get(key))}" for key in ("temporal_score","appearance_score","action_score","person_object_score","carried_object_score"))
            tracks=row.get("appearance_action_track_match") if m=="T" else None
            rank=", ".join(v.get("fusion_components",[])) or "ranking inputs unavailable"
            if tracks:
                rank += f"; appearance ID {tracks['appearance_track_id']}, action ID {tracks['action_track_id']} (predicted IDs)"
                if tracks["different_tracks"]: rank += "; <span class='flag'>different tracks supplied appearance/action maxima</span>"
            report.append(f"<tr><td>{m}</td><td>{interval}</td><td>{_fmt(v.get('temporal_iou'))}</td><td class='scores'>{html.escape(scores)}</td><td>{rank}</td></tr>")
        report.append("</table><h3>Selected evidence crops</h3><div class='crops'>")
        for crop in row["crops"]:
            track=format_score_track_label(crop["score_label"],crop.get("score"),crop.get("track_id"),False)
            img=f"<img src='assets/{html.escape(crop['path'])}' alt='evidence crop'>" if crop.get("path") else "<p>Crop image missing</p>"
            report.append(f"<div class='crop'>{img}<p>{html.escape(crop['method'])} · {html.escape(crop['role'])}</p><p>{html.escape(track)}</p><p>source t={crop['timestamp_seconds']:.2f}s · frame {crop['frame_index']}</p></div>")
        report.append("</div><p class='warn'>Appearance/action track match: "+(
            f"<span class='flag'>different predicted tracks ({row['appearance_action_track_match']['appearance_track_id']} vs {row['appearance_action_track_match']['action_track_id']})</span>" if row.get("appearance_action_track_match",{}).get("different_tracks") else "same predicted track or unavailable")+". Identity accuracy is unmeasured.</p></section>")
    report.append("<footer>Video frames show saved YOLO11/ByteTrack boxes only. Query/method frames are interval midpoints; timeline shows complete saved intervals. Scores are copied from saved results and retain missing values as missing.</footer></body></html>")
    (output / "index.html").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps({"output":str(output.resolve()),"queries":len(review_rows),"video_status":video_status,
        "missing_count":len(missing),"missing":missing},indent=2))


if __name__ == "__main__":
    main()
