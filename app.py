"""
app.py — Streamlit UI for the Moment-DETR baseline.

Extends the existing, working Moment-DETR pipeline (run_on_video/run.py)
with real video clip extraction and an interactive UI. Does NOT modify,
retrain, or replace the Moment-DETR model in any way — it only consumes
the structured predictions that MomentDETRPredictor.localize_moment()
already returns.

Run with:
    streamlit run app.py
(run from the repository root, same place you'd run run_on_video/run.py)
"""

import math
import os
import sys
import time
import uuid
import zipfile
from pathlib import Path

import streamlit as st
import torch

# Ensure repo root is importable regardless of how streamlit was launched
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_on_video.run import MomentDETRPredictor  # noqa: E402  (existing, unmodified)
from extension.query_logic import (
    ABLATION_COMPONENTS,
    fuse_candidate_ranks,
    parse_retrieval_query,
)
from run_on_video.clip_extractor import (
    get_video_duration,
    is_within_duration_limit,
    normalize_and_validate_scored,
    extract_clip,
    format_timestamp,
)
from extension.ui_person_visualization import (
    build_person_crop_gallery_items,
    crop_person_detections,
    draw_person_boxes,
    midpoint_timestamp,
    person_detections_from_result,
    representative_frame_timestamps,
)

CKPT_PATH = "run_on_video/moment_detr_ckpt/model_best.ckpt"
OUTPUT_ROOT = "outputs"
MAX_VIDEO_SECONDS = 150  # model accepts at most 75 two-second segments
RANKING_LABELS = {
    "temporal": "A. Temporal only (baseline)",
    "temporal_appearance": "B. Temporal + appearance",
    "temporal_action": "C. Temporal + action",
    "temporal_appearance_action": "D. Temporal + appearance + action",
    "temporal_appearance_action_carried_object": "E. Temporal + appearance + action + detected object",
    "temporal_appearance_person_object_carried_object": "Research: temporal + person-crop appearance/object evidence",
}
YOLO_WEIGHTS_PATH = Path(__file__).resolve().parent / "yolo11n.pt"


st.set_page_config(page_title="Moment-DETR Video Moment Retrieval", layout="wide")


@st.cache_resource(show_spinner="Loading Moment-DETR model (first run only)...")
def load_predictor():
    """Loaded once per server process and cached across reruns/sessions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = MomentDETRPredictor(
        ckpt_path=CKPT_PATH,
        clip_model_name_or_path="ViT-B/32",
        device=device,
    )
    return predictor, device


@st.cache_resource(show_spinner=False)
def load_person_visualization_detector():
    """Reuse the scoring detector when available, otherwise load local YOLO once."""
    matching_module = sys.modules.get("extension.attribute_matching")
    existing_detector = getattr(matching_module, "yolo_model", None)
    if existing_detector is not None:
        return existing_detector
    if not YOLO_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Local YOLO weights are missing: {YOLO_WEIGHTS_PATH}")
    from ultralytics import YOLO

    detector = YOLO(str(YOLO_WEIGHTS_PATH))
    if matching_module is not None:
        matching_module.yolo_model = detector
    return detector


@st.cache_data(max_entries=8, show_spinner=False)
def detect_people_for_ui(video_path, timestamp_seconds, video_mtime_ns, weights_mtime_ns):
    """Return one cached source frame and its local YOLO person detections."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the uploaded video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not decode a frame at {timestamp_seconds:.2f}s")

    detector = load_person_visualization_detector()
    results = detector.predict(source=frame_bgr, classes=[0], verbose=False)
    result = results[0] if results else None
    detections = person_detections_from_result(result) if result is not None else []
    return frame_bgr, detections


def new_session_dir():
    session_id = uuid.uuid4().hex[:10]
    session_dir = os.path.join(OUTPUT_ROOT, f"session_{session_id}")
    os.makedirs(os.path.join(session_dir, "clips"), exist_ok=True)
    return session_dir


def run_pipeline(uploaded_video, query_text, padding, ranking_mode="temporal_appearance_action"):
    """Runs the full pipeline ONCE: save upload -> inference -> rerank -> validate -> extract.

    Returns a result dict on success, or None on a fatal error (with the
    error already surfaced to the user via the status widget).
    """
    pipeline_started = time.perf_counter()
    timings = {}
    session_dir = new_session_dir()
    video_path = os.path.join(session_dir, "original_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    status = st.status("Running pipeline...", expanded=True)
    status.write("✅ Video uploaded")

    try:
        stage_started = time.perf_counter()
        duration = get_video_duration(video_path)
        timings["video_probe_seconds"] = time.perf_counter() - stage_started
    except Exception as e:
        status.update(label="Failed to read video", state="error")
        status.write(f"Could not read video metadata: {e}")
        return None
    status.write(f"✅ Video duration: {duration:.2f}s")

    if not is_within_duration_limit(duration, MAX_VIDEO_SECONDS):
        status.write(
            f"⚠️ This video is {duration:.1f}s long. The pretrained checkpoint's "
            f"positional embeddings only support approximately {MAX_VIDEO_SECONDS}s "
            f"({MAX_VIDEO_SECONDS // 2} x 2-second clips). Please upload a shorter clip."
        )
        status.update(label="Video too long for this checkpoint", state="error")
        return None

    try:
        stage_started = time.perf_counter()
        predictor, device = load_predictor()
        timings["model_load_seconds"] = time.perf_counter() - stage_started
        status.write(f"✅ Model loaded ({device})")
    except Exception as e:
        status.update(label="Model load failed", state="error")
        status.write(str(e))
        return None

    query_parts = parse_retrieval_query(query_text)
    status.write("⏳ Running Moment-DETR inference on the original query...")
    try:
        stage_started = time.perf_counter()
        # Keep the full original query for temporal grounding.
        raw_predictions = predictor.localize_moment(
            video_path=video_path, query_list=[query_text]
        )
        timings["temporal_inference_seconds"] = time.perf_counter() - stage_started
    except Exception as e:
        status.update(label="Inference failed", state="error")
        status.write(str(e))
        return None
    status.write("✅ Moment-DETR inference complete")

    # --- convert raw [start, end, score] triples into the dict format
    #     rerank_with_attributes() expects ---
    raw_windows = raw_predictions[0]["pred_relevant_windows"]
    candidates = [
        {"start": w[0], "end": w[1], "temporal_score": w[2]}
        for w in raw_windows
    ]

    requested_components = ABLATION_COMPONENTS[ranking_mode]
    appearance_text = (
        query_parts.appearance_text if "appearance" in requested_components else None
    )
    action_text = query_parts.action_text if "action" in requested_components else None
    carried_object_texts = (
        query_parts.carried_object_texts if "carried_object" in requested_components else ()
    )
    detector_object_requests = (
        tuple(dict.fromkeys((*query_parts.carried_object_texts, *query_parts.unresolved_carried_object_texts)))
        if "carried_object" in requested_components else ()
    )
    carried_object_mentions = (
        query_parts.carried_object_mentions if "person_object" in requested_components else ()
    )
    if appearance_text or action_text or detector_object_requests or carried_object_mentions:
        status.write("Scoring semantic cues separately from temporal confidence...")
        try:
            stage_started = time.perf_counter()
            # Lazy import keeps the temporal-only baseline independent of the
            # YOLO/OpenCLIP person-appearance and frame-semantic models.
            from extension.attribute_matching import score_candidates_with_semantics

            scored_candidates = score_candidates_with_semantics(
                video_path,
                candidates,
                appearance_text=appearance_text,
                action_text=action_text,
                carried_object_texts=detector_object_requests,
                carried_object_mentions=carried_object_mentions,
            )
            timings["semantic_scoring_seconds"] = time.perf_counter() - stage_started
        except Exception as e:
            status.update(label="Semantic scoring failed", state="error")
            status.write(str(e))
            return None
    else:
        timings["semantic_scoring_seconds"] = 0.0
        scored_candidates = [
            {
                **candidate,
                "appearance_score": None,
                "attribute_score": None,
                "appearance_evidence_status": "not_requested",
                "action_score": None,
                "action_evidence_status": "not_requested",
                "carried_object_score": None,
                "carried_object_detections": [],
                "carried_object_evidence_status": "not_requested",
                "carried_object_evidence_states": {},
                "person_object_score": None,
                "person_object_evidence_status": "not_requested",
                "person_object_mention_scores": [],
            }
            for candidate in candidates
        ]
        status.write(
            "No requested semantic cue was recognized; using temporal candidate scores."
        )

    stage_started = time.perf_counter()
    ranked_candidates = fuse_candidate_ranks(scored_candidates, ranking_mode)

    normalized = normalize_and_validate_scored(ranked_candidates, duration, padding=padding)
    timings["ranking_validation_seconds"] = time.perf_counter() - stage_started
    n_valid = sum(1 for r in normalized if r["valid"])
    status.write(
        f"✅ Ranked {len(normalized)} candidates ({n_valid} valid)"
    )

    status.write("⏳ Extracting clips...")
    stage_started = time.perf_counter()
    clips_dir = os.path.join(session_dir, "clips")
    for r in normalized:
        if not r["valid"]:
            r["clip_path"] = None
            r["extraction_error"] = r["reason"]
            continue
        ok, result = extract_clip(
            video_path, r["padded_start"], r["padded_end"], clips_dir, r["index"]
        )
        if ok:
            r["clip_path"] = result
            r["extraction_error"] = None
        else:
            r["clip_path"] = None
            r["extraction_error"] = result

    n_extracted = sum(1 for r in normalized if r.get("clip_path"))
    status.write(
        f"✅ Clip extraction complete: {n_extracted}/{len(normalized)} clips generated"
    )
    timings["clip_extraction_seconds"] = time.perf_counter() - stage_started
    timings["total_pipeline_seconds"] = time.perf_counter() - pipeline_started
    status.update(label="Pipeline complete", state="complete")

    return dict(
        session_dir=session_dir,
        video_path=video_path,
        video_duration=duration,
        predictions=normalized,
        query=query_text,
        appearance_text=appearance_text,
        action_text=action_text,
        person_text=query_parts.person_text,
        object_texts=query_parts.object_texts,
        carried_object_texts=carried_object_texts,
        carried_object_mentions=carried_object_mentions,
        ranking_mode=ranking_mode,
        scored_candidates=scored_candidates,
        device=device,
        stage_timings=timings,
        model_eval_mode=not predictor.model.training,
        temporal_candidates=candidates,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎥 Natural Language Video Moment Retrieval")
st.caption(
    "Moment-DETR baseline — upload a video, describe a moment, get the matching clip(s)."
)

for key, default in [
    ("session_dir", None),
    ("source_video_path", None),
    ("predictions", None),
    ("video_duration", None),
    ("query", None),
    ("appearance_text", None),
    ("action_text", None),
    ("person_text", None),
    ("object_texts", None),
    ("carried_object_texts", None),
    ("carried_object_mentions", None),
    ("ranking_mode", None),
    ("zip_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Confidence threshold (filters displayed results)", 0.0, 1.0, 0.0, 0.01
    )
    padding = st.number_input(
        "Padding around detected moment (seconds)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )
    ranking_mode = st.selectbox(
        "Candidate ranking ablation",
        options=list(RANKING_LABELS),
        format_func=lambda mode: RANKING_LABELS[mode],
        index=3,
        help=(
            "Semantic modes fuse within-query component rank percentiles. "
            "Missing cues are omitted; temporal-only remains available as baseline."
        ),
    )
    st.caption(f"Note: this checkpoint supports videos up to {MAX_VIDEO_SECONDS}s.")

uploaded_video = st.file_uploader(
    "Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"]
)
query_text = st.text_input(
    "Describe the moment you're looking for",
    placeholder="e.g. Chef makes pizza and cuts it up.",
)

if uploaded_video is not None:
    st.video(uploaded_video)
    st.caption(
        f"File: {uploaded_video.name} — {uploaded_video.size / (1024 * 1024):.2f} MB"
    )

analyze_clicked = st.button(
    "Analyze Video",
    type="primary",
    disabled=(uploaded_video is None or not query_text.strip()),
)

if analyze_clicked and uploaded_video is not None and query_text.strip():
    result = run_pipeline(uploaded_video, query_text, padding, ranking_mode)
    if result is not None:
        st.session_state.session_dir = result["session_dir"]
        st.session_state.source_video_path = result["video_path"]
        st.session_state.video_duration = result["video_duration"]
        st.session_state.predictions = result["predictions"]
        st.session_state.query = result["query"]
        st.session_state.appearance_text = result["appearance_text"]
        st.session_state.action_text = result["action_text"]
        st.session_state.person_text = result["person_text"]
        st.session_state.object_texts = result["object_texts"]
        st.session_state.carried_object_texts = result["carried_object_texts"]
        st.session_state.carried_object_mentions = result["carried_object_mentions"]
        st.session_state.ranking_mode = result["ranking_mode"]
        st.session_state.zip_path = None  # invalidate any previous zip

# ---------------------------------------------------------------------------
# Results (rendered from session_state so reruns from download clicks etc.
# do NOT re-trigger inference or re-extraction)
# ---------------------------------------------------------------------------
if st.session_state.predictions:
    st.divider()
    st.subheader(f"Results for: \u201c{st.session_state.query}\u201d")

    preds = st.session_state.predictions
    shown = [p for p in preds if p["score"] >= confidence_threshold]

    if not shown:
        st.info(
            "No relevant moments were detected above the current confidence threshold."
        )
    else:
        for p in shown:
            with st.container(border=True):
                st.markdown(f"**Rank {p['index']} — {RANKING_LABELS.get(p.get('ranking_method'), 'Temporal ranking')}**")
                st.caption(
                    f"Temporal confidence: {p['score']:.4f}; "
                    f"normalized rank fusion: {p['fused_score']:.4f}"
                    if p.get("fused_score") is not None
                    else f"Temporal confidence: {p['score']:.4f}"
                )
                if p.get("attribute_score") is not None:
                    st.caption(
                        "Best person-crop OpenCLIP cosine similarity (raw, uncalibrated): "
                        f"{p['attribute_score']:.4f}"
                    )
                elif st.session_state.get("appearance_text"):
                    st.caption(
                        "Person-appearance evidence: "
                        f"{p.get('appearance_evidence_status', 'unavailable')} (raw OpenCLIP cosine is uncalibrated)."
                    )
                if p.get("action_score") is not None:
                    st.caption(
                        "Mean sampled-frame/action OpenCLIP cosine similarity: "
                        f"{p['action_score']:.4f} (static-frame semantic evidence, not motion recognition)"
                    )
                if p.get("person_object_score") is not None:
                    st.caption(
                        "Best single-person-crop object phrase OpenCLIP cosine similarity: "
                        f"{p['person_object_score']:.4f} (raw and uncalibrated; does not establish identity or carrying)."
                    )
                elif st.session_state.get("carried_object_mentions") and p.get("person_object_evidence_status") not in (None, "not_requested"):
                    st.caption(
                        "Person-crop object phrase evidence: "
                        f"{p.get('person_object_evidence_status')} (unavailable evidence is omitted from ranking)."
                    )
                if p.get("carried_object_score") is not None:
                    detected = ", ".join(
                        f"{item['class_name']} ({item['confidence']:.2f})"
                        for item in p.get("carried_object_detections", [])
                    )
                    st.caption(
                        f"YOLO object detections (raw confidence, not calibrated and not proof of carrying): {detected}; "
                        f"candidate object score: {p['carried_object_score']:.4f}"
                    )
                elif p.get("carried_object_evidence_status") == "no_matching_detection":
                    st.caption(
                        "No requested COCO object was detected in sampled frames; this is missing visual evidence, not a confirmed mismatch."
                    )
                elif st.session_state.get("carried_object_texts") and p.get("carried_object_evidence_status") not in (None, "not_requested"):
                    st.caption(
                        "Carried-object category evidence unavailable: "
                        f"{p.get('carried_object_evidence_status')} (object detection does not establish who carried it)."
                    )
                st.write(
                    f"Detected timestamp: `{format_timestamp(p['start'])} → {format_timestamp(p['end'])}` "
                    f"({p['duration']:.2f}s)"
                )
                if padding > 0 and p["valid"]:
                    st.caption(
                        f"Extracted range (with {padding:.1f}s padding): "
                        f"{format_timestamp(p['padded_start'])} → {format_timestamp(p['padded_end'])}"
                    )

                if p.get("clip_path"):
                    st.video(p["clip_path"])
                    with open(p["clip_path"], "rb") as f:
                        st.download_button(
                            f"⬇ Download Clip {p['index']}",
                            data=f.read(),
                            file_name=os.path.basename(p["clip_path"]),
                            mime="video/mp4",
                            key=f"dl_{p['index']}",
                        )
                elif not p["valid"]:
                    st.warning(f"Skipped — invalid prediction ({p['reason']}).")
                else:
                    st.error(
                        f"Clip {p['index']} could not be extracted: {p.get('extraction_error')}"
                    )

        ok_clips = [p for p in shown if p.get("clip_path")]
        if ok_clips:
            st.divider()
            st.subheader("Inspect a ranked candidate")
            clip_by_rank = {p["index"]: p for p in ok_clips}
            session_key = os.path.basename(st.session_state.session_dir or "session")
            selected_rank = st.selectbox(
                "Selected clip for evidence and person-box view",
                options=list(clip_by_rank),
                format_func=lambda rank: (
                    f"Rank {rank} · {format_timestamp(clip_by_rank[rank]['start'])}–"
                    f"{format_timestamp(clip_by_rank[rank]['end'])}"
                ),
                key=f"inspect_rank_{session_key}",
            )
            selected = clip_by_rank[selected_rank]
            with st.container(border=True):
                st.markdown(f"**Rank {selected['index']} selected interval**")
                st.write(
                    f"`{format_timestamp(selected['start'])} → "
                    f"{format_timestamp(selected['end'])}` "
                    f"({selected['duration']:.2f}s)"
                )

                def display_score(value, status=None):
                    if value is not None:
                        try:
                            numeric_value = float(value)
                            if math.isfinite(numeric_value):
                                return f"{numeric_value:.4f}"
                            return "Unavailable (non-finite value)"
                        except (TypeError, ValueError):
                            return "Unavailable (invalid saved value)"
                    if status and status != "not_requested":
                        return f"Unavailable ({status})"
                    return "Not computed for this ranking"

                evidence_rows = [
                    {
                        "Evidence": "Moment-DETR temporal confidence",
                        "Saved value": display_score(
                            selected.get("temporal_score", selected.get("score"))
                        ),
                        "Meaning": "Proposal confidence; not a calibrated probability.",
                    },
                    {
                        "Evidence": "Within-query rank-fusion score",
                        "Saved value": display_score(selected.get("fused_score")),
                        "Meaning": "Mean of available normalized evidence ranks.",
                    },
                    {
                        "Evidence": "Person-crop appearance OpenCLIP similarity",
                        "Saved value": display_score(
                            selected.get("appearance_score", selected.get("attribute_score")),
                            selected.get("appearance_evidence_status"),
                        ),
                        "Meaning": "Raw, uncalibrated visual-text similarity; not identity verification.",
                    },
                    {
                        "Evidence": "Full-frame action-phrase OpenCLIP similarity",
                        "Saved value": display_score(selected.get("action_score"), selected.get("action_evidence_status")),
                        "Meaning": "Static-frame semantic evidence; not action recognition.",
                    },
                    {
                        "Evidence": "Person-crop object-phrase OpenCLIP similarity",
                        "Saved value": display_score(
                            selected.get("person_object_score"),
                            selected.get("person_object_evidence_status"),
                        ),
                        "Meaning": "Phrase similarity for a detected person crop; not identity or ownership proof.",
                    },
                    {
                        "Evidence": "YOLO requested object-category score",
                        "Saved value": display_score(
                            selected.get("carried_object_score"),
                            selected.get("carried_object_evidence_status"),
                        ),
                        "Meaning": "Category detections do not establish which person owns or carries the object.",
                    },
                ]
                st.table(evidence_rows)

                show_person_boxes = st.checkbox(
                    "Show YOLO person boxes on one midpoint frame",
                    value=False,
                    key=f"show_person_boxes_{session_key}",
                    help=(
                        "Runs the existing local YOLO11n detector on one frame at the "
                        "selected interval midpoint. Boxes show detected people, not a verified query match."
                    ),
                )
                if show_person_boxes:
                    source_video = st.session_state.get("source_video_path") or os.path.join(
                        st.session_state.session_dir or "", "original_video.mp4"
                    )
                    if not os.path.isfile(source_video):
                        st.warning(
                            "Person visualization is unavailable because the saved source video "
                            "for this session could not be found. No frame was substituted."
                        )
                    else:
                        try:
                            frame_timestamp = midpoint_timestamp(
                                selected["start"], selected["end"]
                            )
                            video_mtime_ns = os.stat(source_video).st_mtime_ns
                            weights_mtime_ns = YOLO_WEIGHTS_PATH.stat().st_mtime_ns
                            with st.spinner("Detecting people in one selected-interval frame..."):
                                frame_bgr, people = detect_people_for_ui(
                                    source_video,
                                    frame_timestamp,
                                    video_mtime_ns,
                                    weights_mtime_ns,
                                )
                            annotated_frame = draw_person_boxes(frame_bgr, people)
                            st.image(
                                annotated_frame,
                                caption=(
                                    f"Source frame at {format_timestamp(frame_timestamp)}. "
                                    "Boxes mark YOLO-detected people only; confidence is detector confidence, "
                                    "not identity confidence or evidence of query match."
                                ),
                                width="stretch",
                            )
                            st.caption(f"YOLO detected {len(people)} person box(es) in this frame.")
                            if not people:
                                st.info("No person detections were returned for this frame.")
                        except Exception as exc:
                            st.warning(f"Person-box visualization could not be produced: {exc}")

                st.markdown("#### Detected person crops")
                display_query = query_text or st.session_state.get("query")
                st.write("**Query:**", display_query or "Unavailable for this session")
                st.caption(
                    "Crops are unverified person detections. Candidate numbers restart for each frame "
                    "and do not indicate persistent identity. The available person/object score is "
                    "aggregated at the proposal level, so it cannot be assigned to an individual crop."
                )
                show_person_crops = st.checkbox(
                    "Show detected person crops across up to 3 frames",
                    value=False,
                    key=f"show_person_crops_{session_key}",
                    help=(
                        "Runs the existing local YOLO11n person detector only on representative "
                        "frames inside the selected candidate interval."
                    ),
                )
                if show_person_crops:
                    source_video = st.session_state.get("source_video_path") or os.path.join(
                        st.session_state.session_dir or "", "original_video.mp4"
                    )
                    if not os.path.isfile(source_video):
                        st.warning(
                            "Person crops are unavailable because the saved source video "
                            "for this session could not be found. No replacement video was used."
                        )
                    elif not YOLO_WEIGHTS_PATH.is_file():
                        st.warning(f"Person crops are unavailable; local YOLO weights are missing: {YOLO_WEIGHTS_PATH}")
                    else:
                        try:
                            frame_timestamps = representative_frame_timestamps(
                                selected["start"], selected["end"], max_frames=3
                            )
                            video_mtime_ns = os.stat(source_video).st_mtime_ns
                            weights_mtime_ns = YOLO_WEIGHTS_PATH.stat().st_mtime_ns
                            gallery_items = []
                            with st.spinner(
                                f"Detecting people in {len(frame_timestamps)} representative frame(s)..."
                            ):
                                for frame_index, frame_timestamp in enumerate(frame_timestamps, start=1):
                                    try:
                                        frame_bgr, people = detect_people_for_ui(
                                            source_video,
                                            frame_timestamp,
                                            video_mtime_ns,
                                            weights_mtime_ns,
                                        )
                                        crops = crop_person_detections(frame_bgr, people)
                                        gallery_items.extend(
                                            build_person_crop_gallery_items(
                                                frame_index, frame_timestamp, crops
                                            )
                                        )
                                    except Exception as exc:
                                        st.warning(
                                            f"Could not produce person crops at "
                                            f"{format_timestamp(frame_timestamp)}: {exc}"
                                        )
                            if not gallery_items:
                                st.info(
                                    "No valid person crops were available in the sampled frames. "
                                    "This can mean no detections, invalid boxes, or frame/detector errors."
                                )
                            else:
                                crop_columns = st.columns(3)
                                for crop_index, item in enumerate(gallery_items):
                                    with crop_columns[crop_index % len(crop_columns)]:
                                        st.image(
                                            item["image_rgb"],
                                            caption=item["label"],
                                            width="stretch",
                                        )
                        except Exception as exc:
                            st.warning(f"Person crop gallery could not be produced: {exc}")

            if st.button("📦 Prepare Download All (ZIP)"):
                zip_path = os.path.join(
                    st.session_state.session_dir, "moment_detr_clips.zip"
                )
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for p in ok_clips:
                        zf.write(p["clip_path"], arcname=f"clip_{p['index']:03d}.mp4")
                st.session_state.zip_path = zip_path

            if st.session_state.zip_path and os.path.exists(st.session_state.zip_path):
                with open(st.session_state.zip_path, "rb") as f:
                    st.download_button(
                        "⬇ Download All Clips (ZIP)",
                        data=f.read(),
                        file_name="moment_detr_clips.zip",
                        mime="application/zip",
                    )
