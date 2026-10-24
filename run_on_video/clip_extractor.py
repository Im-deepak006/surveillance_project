"""
run_on_video/clip_extractor.py

Extension layer for the Moment-DETR baseline: turns predicted
[start, end, score] windows into real, playable MP4 clips.

This module does NOT touch the model, the CLIP feature extractor, or
the prediction logic in run_on_video/run.py in any way. It only
consumes the `pred_relevant_windows` output that MomentDETRPredictor
already produces.

Uses ffmpeg-python, which is already a project dependency (see
run_on_video/data_utils.py), so no new heavyweight dependency is added.
"""
import os
import uuid
import math
import ffmpeg


def is_within_duration_limit(duration: float, limit: float, tolerance: float = 0.1) -> bool:
    """Accept small container-duration rounding while rejecting invalid values."""
    try:
        duration = float(duration)
        limit = float(limit)
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(duration)
        and math.isfinite(limit)
        and math.isfinite(tolerance)
        and limit >= 0
        and tolerance >= 0
        and duration <= limit + tolerance
    )


def get_video_duration(video_path: str) -> float:
    """Return the duration of a video file in seconds via ffprobe.

    Raises ValueError if duration cannot be determined.
    """
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is not None and video_stream.get("duration"):
        return float(video_stream["duration"])
    fmt_duration = probe.get("format", {}).get("duration")
    if fmt_duration:
        return float(fmt_duration)
    raise ValueError(f"Could not determine duration for video: {video_path}")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.cc, or HH:MM:SS.cc for videos >= 1 hour.

    e.g. 55.0335 -> '00:55.03', 67.2583 -> '01:07.26'
    """
    if seconds is None:
        return "--:--"
    seconds = max(0.0, seconds)
    total_centiseconds = int(round(seconds * 100))
    hours, rem = divmod(total_centiseconds, 100 * 3600)
    minutes, rem = divmod(rem, 100 * 60)
    secs, centis = divmod(rem, 100)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centis:02d}"
    return f"{minutes:02d}:{secs:02d}.{centis:02d}"


def normalize_and_validate(raw_windows, video_duration, padding=0.0):
    """
    Convert raw Moment-DETR predictions into a validated, sorted list of
    prediction dicts. Invalid entries are KEPT (marked valid=False) so
    the caller can report them rather than silently dropping data.

    Args:
        raw_windows: list of [start, end, score] (as returned by
            MomentDETRPredictor.localize_moment(...)[i]['pred_relevant_windows'])
        video_duration: float, total video duration in seconds
        padding: float, seconds of padding to add around each valid
            window before extraction (default 0 = no padding)

    Returns:
        List[dict] sorted by score descending, each with keys:
        index, start, end, score, duration, padded_start, padded_end,
        valid, reason
    """
    results = []
    for row in raw_windows:
        start, end, score = float(row[0]), float(row[1]), float(row[2])
        valid = True
        reason = None

        if start < 0:
            start = 0.0

        if end <= start:
            valid = False
            reason = "end <= start"
        elif start >= video_duration:
            valid = False
            reason = "start is beyond video duration"
        else:
            if end > video_duration:
                end = video_duration  # clamp

        if valid:
            padded_start = max(0.0, start - padding)
            padded_end = min(video_duration, end + padding)
        else:
            padded_start, padded_end = start, end

        results.append(dict(
            start=round(start, 4),
            end=round(end, 4),
            score=round(score, 4),
            duration=round(end - start, 4) if valid else 0.0,
            padded_start=round(padded_start, 4),
            padded_end=round(padded_end, 4),
            valid=valid,
            reason=reason,
        ))

    # sort by confidence score, descending (consistent with existing
    # model output ordering — see run_on_video/run.py sorted() call)
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results):
        r["index"] = i + 1

    return results

def normalize_and_validate_scored(scored_candidates, video_duration, padding=0.0):
    """
    Consume candidate dicts and preserve their separate temporal, appearance,
    action, carried-object, and person-crop evidence through timestamp validation. The
    selected fused ranking controls order; `score` remains temporal confidence.
    """
    results = []
    for cand in scored_candidates:
        start, end = float(cand["start"]), float(cand["end"])
        valid, reason = True, None

        if start < 0:
            start = 0.0
        if end <= start:
            valid, reason = False, "end <= start"
        elif start >= video_duration:
            valid, reason = False, "start is beyond video duration"
        else:
            end = min(end, video_duration)

        if valid:
            padded_start = max(0.0, start - padding)
            padded_end = min(video_duration, end + padding)
        else:
            padded_start, padded_end = start, end

        results.append(dict(
            start=round(start, 4),
            end=round(end, 4),
            score=round(cand["temporal_score"], 4),
            temporal_score=round(cand["temporal_score"], 4),
            attribute_score=(
                round(cand["attribute_score"], 4)
                if cand.get("attribute_score") is not None
                else None
            ),
            appearance_evidence_status=cand.get("appearance_evidence_status", "not_requested"),
            appearance_score=(
                round(cand["appearance_score"], 4)
                if cand.get("appearance_score") is not None
                else (
                    round(cand["attribute_score"], 4)
                    if cand.get("attribute_score") is not None
                    else None
                )
            ),
            action_score=(
                round(cand["action_score"], 4)
                if cand.get("action_score") is not None
                else None
            ),
            action_evidence_status=cand.get("action_evidence_status", "not_requested"),
            temporal_rank_score=cand.get("temporal_rank_score"),
            appearance_rank_score=cand.get("appearance_rank_score"),
            action_rank_score=cand.get("action_rank_score"),
            carried_object_score=(
                round(cand["carried_object_score"], 4)
                if cand.get("carried_object_score") is not None
                else None
            ),
            carried_object_rank_score=cand.get("carried_object_rank_score"),
            person_object_score=(
                round(cand["person_object_score"], 4)
                if cand.get("person_object_score") is not None
                else None
            ),
            person_object_rank_score=cand.get("person_object_rank_score"),
            person_object_evidence_status=cand.get("person_object_evidence_status", "not_requested"),
            person_object_mention_scores=cand.get("person_object_mention_scores", []),
            carried_object_detections=cand.get("carried_object_detections", []),
            carried_object_evidence_status=cand.get("carried_object_evidence_status", "not_requested"),
            carried_object_evidence_states=cand.get("carried_object_evidence_states", {}),
            fused_score=cand.get("fused_score"),
            fusion_components=cand.get("fusion_components", []),
            ranking_method=cand.get("ranking_method", "temporal"),
            duration=round(end - start, 4) if valid else 0.0,
            padded_start=round(padded_start, 4),
            padded_end=round(padded_end, 4),
            valid=valid,
            reason=reason,
        ))

    # Preserve the selected ranking method's ordering; temporal confidence
    # remains separately available in `score` and `temporal_score`.
    results.sort(
        key=lambda r: (
            r["fused_score"] is not None,
            r["fused_score"] if r["fused_score"] is not None else r["score"],
        ),
        reverse=True,
    )
    for i, r in enumerate(results):
        r["index"] = i + 1
    return results

def extract_clip(video_path: str, start: float, end: float, output_dir: str, index: int):
    """
    Extract [start, end] seconds from video_path into a new MP4 file
    under output_dir. Does not modify or overwrite the source video.

    Returns:
        (True, output_path) on success
        (False, error_message) on failure
    """
    if end <= start:
        return False, "end <= start"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"clip_{index:03d}_{uuid.uuid4().hex[:6]}.mp4")

    try:
        (
            ffmpeg
            .input(video_path, ss=start, t=end - start)
            .output(
                out_path,
                vcodec="libx264",
                acodec="aac",
                avoid_negative_ts="make_zero",
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        return False, stderr.strip().splitlines()[-1] if stderr.strip() else "ffmpeg extraction failed"
    except Exception as e:
        return False, str(e)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False, "ffmpeg produced no output file"

    return True, out_path
