"""Helpers for optional person-box and unverified crop diagnostics in the UI."""

import math


def midpoint_timestamp(start, end):
    """Return the interval midpoint in seconds, rejecting invalid intervals."""
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        raise ValueError("Interval boundaries must be numeric")
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Interval must have finite boundaries with end > start")
    return (start + end) / 2.0


def representative_frame_timestamps(start, end, max_frames=3):
    """Choose evenly spaced interior sample times for a selected interval."""
    try:
        start = float(start)
        end = float(end)
        max_frames = int(max_frames)
    except (TypeError, ValueError):
        raise ValueError("Interval boundaries and frame count must be numeric")
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Interval must have finite boundaries with end > start")
    if max_frames < 1:
        raise ValueError("At least one representative frame is required")
    count = min(max_frames, max(1, int(math.ceil(end - start))))
    # Interior quantiles avoid repeatedly sampling the exact interval edges.
    return [start + (end - start) * (i + 1) / (count + 1) for i in range(count)]


def _to_list(value):
    """Convert common Ultralytics tensor/array fields into Python lists."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def person_detections_from_result(result, confidence_threshold=0.0):
    """Extract class-0 YOLO person boxes without assigning identity labels."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy_rows = _to_list(getattr(boxes, "xyxy", [])) or []
    class_ids = _to_list(getattr(boxes, "cls", [])) or []
    confidences = _to_list(getattr(boxes, "conf", [])) or []
    detections = []
    for xyxy, class_id, confidence in zip(xyxy_rows, class_ids, confidences):
        try:
            class_id = int(class_id)
            confidence = float(confidence)
            coords = [float(value) for value in xyxy]
        except (TypeError, ValueError):
            continue
        if class_id != 0 or not math.isfinite(confidence) or confidence < confidence_threshold:
            continue
        if len(coords) != 4 or not all(math.isfinite(value) for value in coords):
            continue
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"xyxy": coords, "confidence": confidence})
    return detections


def person_detection_label(confidence):
    """Label detector confidence explicitly; it is not an identity score."""
    value = float(confidence)
    if not math.isfinite(value):
        raise ValueError("Detection confidence must be finite")
    return f"Person | YOLO conf {value:.2f}"


def draw_person_boxes(frame_bgr, detections):
    """Draw detected-person boxes and return an RGB image for Streamlit."""
    import cv2

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise ValueError("A readable video frame is required")
    frame = frame_bgr.copy()
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        x1 = max(0, min(width - 1, int(round(x1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y1 = max(0, min(height - 1, int(round(y1))))
        y2 = max(0, min(height - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        color = (44, 190, 132)  # BGR green
        label = person_detection_label(detection["confidence"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 8), min(height - 1, label_top + text_height + baseline + 6)),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, min(height - 1, label_top + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (15, 25, 30),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_person_detections(frame_bgr, detections):
    """Return safely clipped RGB crops for valid person detections.

    Invalid, non-finite, empty, and fully out-of-frame boxes are skipped. The
    candidate number is local to this frame and has no identity semantics.
    """
    import cv2

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise ValueError("A readable video frame is required")
    if getattr(frame_bgr, "ndim", 0) < 2:
        raise ValueError("Frame must have image dimensions")
    height, width = frame_bgr.shape[:2]
    items = []
    for detection in detections or []:
        try:
            coords = [float(value) for value in detection["xyxy"]]
            confidence = float(detection["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
            continue
        if not math.isfinite(confidence):
            continue
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        left = max(0, min(width, math.floor(x1)))
        top = max(0, min(height, math.floor(y1)))
        right = max(0, min(width, math.ceil(x2)))
        bottom = max(0, min(height, math.ceil(y2)))
        if right <= left or bottom <= top:
            continue
        crop = frame_bgr[top:bottom, left:right]
        if crop.size == 0:
            continue
        items.append({
            "candidate_number": len(items) + 1,
            "confidence": confidence,
            "xyxy_clipped": [left, top, right, bottom],
            "image_rgb": cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2RGB),
            "verification_status": "unverified",
        })
    return items


def build_person_crop_gallery_items(frame_index, timestamp_seconds, crops):
    """Attach frame/time metadata and explicit unverified labels to crops."""
    try:
        frame_index = int(frame_index)
        timestamp_seconds = float(timestamp_seconds)
    except (TypeError, ValueError):
        raise ValueError("Frame index and timestamp must be numeric")
    if frame_index < 1 or not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
        raise ValueError("Frame index and timestamp are invalid")
    return [
        {
            **crop,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "verification_status": "unverified",
            "label": (
                f"Person candidate {crop['candidate_number']} · frame {frame_index} · "
                f"{timestamp_seconds:.2f}s · YOLO conf {float(crop['confidence']):.2f} · unverified"
            ),
        }
        for crop in crops or []
    ]
