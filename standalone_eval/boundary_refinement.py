"""Pure query-aware boundary decision helpers for the isolated pilot."""
from __future__ import annotations

import math


def consensus_boundary_shift(role: str, inside_appearance: float | None,
                             outside_appearance: float | None,
                             inside_action: float | None,
                             outside_action: float | None) -> tuple[float, str]:
    """Apply strict two-cue consensus, returning signed boundary displacement."""
    if role not in {"start", "end"}:
        raise ValueError("role must be 'start' or 'end'")
    values = (inside_appearance, outside_appearance, inside_action, outside_action)
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return 0.0, "rejected_missing_evidence"
    app = 1 if inside_appearance > outside_appearance else -1 if inside_appearance < outside_appearance else 0
    act = 1 if inside_action > outside_action else -1 if inside_action < outside_action else 0
    if app == 0 or act == 0:
        return 0.0, "unchanged_tie"
    if app != act:
        return 0.0, "rejected_conflicting_evidence"
    if role == "start":
        return (0.5, "moved_later") if app > 0 else (-0.5, "moved_earlier")
    return (-0.5, "moved_earlier") if app > 0 else (0.5, "moved_later")


def refine_interval(start: float, end: float, start_shift: float, end_shift: float,
                    duration: float) -> tuple[float, float, str]:
    """Apply shifts, clip to video bounds, and restore invalid results."""
    original = (float(start), float(end))
    if not (math.isfinite(duration) and duration > 0 and
            all(math.isfinite(x) for x in (*original, start_shift, end_shift)) and
            0 <= original[0] < original[1] <= duration):
        return original[0], original[1], "invalid_original_fallback"
    new_start = min(max(original[0] + start_shift, 0.0), duration)
    new_end = min(max(original[1] + end_shift, 0.0), duration)
    if not 0 <= new_start < new_end <= duration:
        return original[0], original[1], "invalid_refinement_fallback"
    return new_start, new_end, "refined"
