import cv2
import math
import torch
import open_clip
from PIL import Image
import numpy as np
from extension.query_logic import (
    fuse_candidate_ranks,
    rank_candidates,
    score_candidate_action,
    score_candidate_appearance_evidence,
    score_candidate_carried_objects_detailed,
    score_person_object_mentions,
    sample_interval_timestamps,
)

# ---------- SETUP (load models once, reuse across all queries) ----------
yolo_model = None
COCO_CARRIED_OBJECT_IDS = {
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "suitcase": 28,
    "bottle": 39,
    "cup": 41,
    "laptop": 63,
    "cell phone": 67,
    "book": 73,
}

# 2. CLIP model for zero-shot attribute matching
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", force_quick_gelu=True
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(device).eval()

# ---------- STEP A: extract frames from a candidate time window ----------

def extract_frames(video_path, start_sec, end_sec, num_frames=5):
    """Pull a handful of evenly-spaced frames from [start_sec, end_sec]."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps = sample_interval_timestamps(start_sec, end_sec, num_frames)
    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)  # BGR, OpenCV format
    cap.release()
    return frames


# ---------- STEP B: detect + crop every person in a frame ----------

def detect_person_and_objects(frame, requested_objects=()):
    """Run one YOLO pass for people and supported requested COCO categories."""
    if frame is None or not getattr(frame, "size", 0):
        return [], []
    global yolo_model
    if yolo_model is None:
        # Action-only queries use the frame-level CLIP scorer and do not load
        # YOLO. The local detector initializes only when appearance evidence
        # is actually requested.
        from ultralytics import YOLO

        yolo_model = YOLO("yolo11n.pt")
    if isinstance(requested_objects, str):
        requested_objects = (requested_objects,)
    targets = tuple(dict.fromkeys(str(value).strip().lower() for value in requested_objects))
    object_ids = [COCO_CARRIED_OBJECT_IDS[name] for name in targets if name in COCO_CARRIED_OBJECT_IDS]
    classes = [0, *object_ids]
    results = yolo_model(frame, classes=classes, verbose=False)
    crops = []
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        class_values = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        names = getattr(result, "names", getattr(yolo_model, "names", {}))
        for box, class_value, confidence in zip(xyxy, class_values, confidences):
            class_id = int(class_value)
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            x1, y1, x2, y2 = map(int, box)
            if class_name == "person":
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
                    continue
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(crop_rgb))
            elif class_name in targets and np.isfinite(confidence):
                detections.append({
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "box_xyxy": [float(value) for value in box],
                })
    return crops, detections


def detect_person_crops(frame):
    """Run YOLO and return usable detected-person crops (PIL images)."""
    return detect_person_and_objects(frame)[0]


def detect_carried_objects(frame, requested_objects):
    """Return local COCO object detections matching requested categories.

    YOLO only reports object categories and confidence. It does not establish
    object ownership or that a nearby person is carrying the object.
    """
    if frame is None or not getattr(frame, "size", 0):
        return []
    return detect_person_and_objects(frame, requested_objects)[1]


# ---------- STEP C: zero-shot attribute scoring with CLIP ----------

def score_attributes(person_crop, attribute_text):
    """
    Given a cropped person image and one appearance description, return the
    raw CLIP cosine similarity. This value is diagnostic and is not on the
    same calibrated scale as Moment-DETR's temporal confidence.
    """
    return score_image_text(person_crop, attribute_text)


def score_image_text(image, text):
    """Return raw OpenCLIP cosine similarity, or None for invalid features."""
    if image is None or not text or not text.strip():
        return None
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_input = clip_tokenizer([text]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features /= text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarity = (image_features @ text_features.T).squeeze()

    value = float(similarity.cpu().item())
    return value if np.isfinite(value) else None


def score_image_text_many(image, texts):
    """Encode one image once and score it against several text phrases."""
    phrases = [text for text in texts if isinstance(text, str) and text.strip()]
    if image is None or not phrases:
        return []
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_input = clip_tokenizer(phrases).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features /= text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        similarities = (image_features @ text_features.T).squeeze(0).cpu().tolist()
    return [float(value) if np.isfinite(value) else None for value in similarities]


def score_action_frame(frame, action_text):
    """Score one BGR frame against an action phrase using image-only OpenCLIP."""
    if frame is None or not getattr(frame, "size", 0):
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return score_image_text(Image.fromarray(rgb), action_text)


def score_candidates_with_semantics(
    video_path,
    candidates,
    appearance_text=None,
    action_text=None,
    appearance_frame_count=15,
    action_frame_count=5,
    carried_object_texts=None,
    carried_object_mentions=None,
):
    """Attach separate raw appearance/action/object evidence per proposal."""
    if (
        not isinstance(appearance_frame_count, int)
        or isinstance(appearance_frame_count, bool)
        or not isinstance(action_frame_count, int)
        or isinstance(action_frame_count, bool)
        or appearance_frame_count < 1
        or action_frame_count < 1
    ):
        raise ValueError("Frame sample counts must be positive integers")

    scored = []
    for candidate in candidates:
        item = {
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
            "person_crop_count_per_frame": [],
        }
        start, end = candidate.get("start"), candidate.get("end")
        try:
            start_value, end_value = float(start), float(end)
            valid_interval = math.isfinite(start_value) and math.isfinite(end_value) and end_value > start_value
        except (TypeError, ValueError):
            valid_interval = False
        if not valid_interval:
            if isinstance(appearance_text, str) and appearance_text.strip():
                item["appearance_evidence_status"] = "invalid_interval"
            if isinstance(action_text, str) and action_text.strip():
                item["action_evidence_status"] = "invalid_interval"
            if carried_object_texts:
                item["carried_object_evidence_status"] = "invalid_interval"
            if carried_object_mentions:
                item["person_object_evidence_status"] = "invalid_interval"
            scored.append(item)
            continue

        if isinstance(carried_object_texts, str):
            requested_objects = (carried_object_texts,)
        else:
            requested_objects = tuple(carried_object_texts or ())
        requested_appearance = isinstance(appearance_text, str) and appearance_text.strip()
        requested_action = isinstance(action_text, str) and action_text.strip()
        if isinstance(carried_object_mentions, str):
            requested_mentions = (carried_object_mentions,)
        else:
            requested_mentions = tuple(carried_object_mentions or ())
        requested_mentions = tuple(
            dict.fromkeys(str(mention).strip() for mention in requested_mentions if str(mention).strip())
        )
        frame_count = max(
            appearance_frame_count if requested_appearance or requested_objects or requested_mentions else 0,
            action_frame_count if requested_action else 0,
        )
        frames = extract_frames(video_path, float(start), float(end), frame_count) if frame_count else []
        frame_evidence_cache = {}

        def get_frame_evidence(frame):
            key = id(frame)
            if key not in frame_evidence_cache:
                frame_evidence_cache[key] = detect_person_and_objects(frame, requested_objects)
            return frame_evidence_cache[key]

        if requested_appearance:
            appearance, appearance_status = score_candidate_appearance_evidence(
                frames,
                appearance_text,
                find_person_crops=lambda frame: get_frame_evidence(frame)[0],
                score_crop=score_attributes,
            )
            item["appearance_score"] = appearance
            item["appearance_evidence_status"] = appearance_status
            # Keep the former field for callers of the existing app/API.
            item["attribute_score"] = appearance

        if requested_appearance or requested_objects or requested_mentions:
            person_crops_by_frame = [get_frame_evidence(frame)[0] for frame in frames]
            item["person_crop_count_per_frame"] = [len(crops) for crops in person_crops_by_frame]
        else:
            person_crops_by_frame = []

        if requested_mentions:
            person_object_score, person_status, mention_scores = score_person_object_mentions(
                (crop for crops in person_crops_by_frame for crop in crops),
                requested_mentions,
                score_attributes,
                score_crop_many=score_image_text_many,
            )
            item["person_object_score"] = person_object_score
            item["person_object_evidence_status"] = person_status
            item["person_object_mention_scores"] = mention_scores

        if requested_action:
            action_frames = frames
            if len(frames) > action_frame_count:
                indices = np.linspace(0, len(frames) - 1, action_frame_count).round().astype(int)
                action_frames = [frames[index] for index in dict.fromkeys(indices.tolist())]
            item["action_score"] = score_candidate_action(
                action_frames,
                action_text,
                score_frame=score_action_frame,
            )
            item["action_evidence_status"] = (
                "scored" if item["action_score"] is not None
                else "no_valid_frame_similarity" if action_frames else "no_frames"
            )

        if requested_objects:
            object_score, detections, status, evidence_states = score_candidate_carried_objects_detailed(
                frames,
                requested_objects,
                detect_objects=lambda frame, targets: [
                    detection for detection in get_frame_evidence(frame)[1]
                    if detection["class_name"] in targets
                ],
            )
            item["carried_object_score"] = object_score
            item["carried_object_detections"] = detections
            item["carried_object_evidence_status"] = status
            item["carried_object_evidence_states"] = evidence_states
        scored.append(item)
    return scored


# ---------- STEP D: tie it all together — re-rank Moment-DETR's candidates ----------

def rerank_with_attributes(video_path, candidates, appearance_text=None, action_text=None, carried_object_texts=None):
    """
    candidates: list of dicts like {"start": 105.0, "end": 116.0, "temporal_score": 0.96}
                — this is literally Moment-DETR's own output format from your earlier run.
    appearance_text: a supported appearance-only phrase parsed from the query.
    The legacy function now delegates to transparent component rank fusion.
    Object detections are category evidence only, not verified person-object
    associations or calibrated probabilities.
    """
    reranked = score_candidates_with_semantics(
        video_path,
        candidates,
        appearance_text=appearance_text,
        action_text=action_text,
        carried_object_texts=carried_object_texts,
    )
    return fuse_candidate_ranks(reranked, "temporal_appearance_action_carried_object")


# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    video_path = "D:\\FYP\\sample.mp4"
    # This is exactly the kind of output your Moment-DETR run already produces:
    moment_detr_candidates = [
        {"start": 105.0, "end": 116.0, "temporal_score": 0.96},
        {"start": 55.0, "end": 67.0, "temporal_score": 0.98},
        {"start": 140.0, "end": 147.0, "temporal_score": 0.76},
    ]

    query_attributes = "person wearing a blue shirt and carrying a backpack"

    final_results = rerank_with_attributes(video_path, moment_detr_candidates, query_attributes)

    for r in final_results:
        print(f"[{r['start']:.1f}s - {r['end']:.1f}s]  "
              f"temporal={r['temporal_score']:.2f}  "
              f"attribute={r['attribute_score'] if r['attribute_score'] is not None else 'not scored'}")
