"""Bounded query decomposition and score-ranking helpers.

The parser deliberately recognizes a small verb and appearance vocabulary. It
is not a general natural-language parser; callers retain the original query
for Moment-DETR whenever a component is unsupported or ambiguous.
"""

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Mapping, Sequence


_PREFIX = re.compile(
    r"^\s*(?:(?:please\s+)?(?:find|search\s+for|look\s+for|locate|show\s+me)\s+)?",
    re.IGNORECASE,
)
_PERSON = re.compile(
    r"\b(?:person|people|someone|somebody|man|woman|individual|chef|cook|worker|officer|guard)\b",
    re.IGNORECASE,
)
_APPEARANCE = re.compile(
    r"\b(?:wearing|wears|dressed\s+in)\b"
    r"|\b(?:shirt|t-?shirt|coat|jacket|trousers|pants|shorts|jeans|dress|hoodie|sweater|vest|blouse|skirt|uniform|hat|cap|shoes|sneakers|socks)\b"
    r"|\b(?:blue|red|green|black|white|yellow|orange|purple|pink|brown|gray|grey)\s+"
    r"(?:shirt|t-?shirt|coat|jacket|trousers|pants|shorts|jeans|dress|hoodie|sweater|vest|blouse|skirt|uniform|hat|cap|shoes|sneakers|socks)\b",
    re.IGNORECASE,
)
_CARRIED_OBJECT_ALIASES = {
    "backpack": "backpack", "backpacks": "backpack",
    "rucksack": "backpack", "rucksacks": "backpack",
    "handbag": "handbag", "handbags": "handbag", "purse": "handbag", "purses": "handbag",
    "shoulder bag": "handbag", "shoulder bags": "handbag",
    "suitcase": "suitcase", "suitcases": "suitcase", "luggage": "suitcase",
    "umbrella": "umbrella", "umbrellas": "umbrella",
    "bottle": "bottle", "bottles": "bottle", "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup",
    "laptop": "laptop", "laptops": "laptop", "phone": "cell phone", "phones": "cell phone",
    "cell phone": "cell phone", "cell phones": "cell phone", "mobile phone": "cell phone", "mobile phones": "cell phone",
    "book": "book", "books": "book",
}
SUPPORTED_CARRIED_OBJECT_CATEGORIES = frozenset({
    "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup",
    "laptop", "cell phone", "book",
})


def sample_interval_timestamps(start_sec: float, end_sec: float, count: int) -> list[float]:
    """Sample an interval evenly; a single sample is its midpoint."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Sample count must be a positive integer")
    start, end = float(start_sec), float(end_sec)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Sample interval must have finite, increasing boundaries")
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]
_CARRIED_OBJECT_PATTERN = "|".join(
    re.escape(name) for name in sorted(_CARRIED_OBJECT_ALIASES, key=len, reverse=True)
)
_CARRIED_MENTION_RE = re.compile(
    r"\b(?:(?:a|an|the|red|gray|grey|bright|yellow|green|blue|black|white|brown|pink|orange|purple|tan|dark|light|pale|small|large|\w+'s)\s+){0,3}(?:"
    + _CARRIED_OBJECT_PATTERN + r")\b",
    re.IGNORECASE,
)
_CARRIED_TRIGGER = re.compile(r"\b(?:carrying|carries|carried|holding|holds|with|wearing|wears)\b", re.IGNORECASE)
_UNSUPPORTED_CARRY_OBJECT_RE = re.compile(
    r"(?:(?P<modifier>(?:\w+\s+){0,2}))(?P<object>(?:shopping|plastic)\s+bags?)\b",
    re.IGNORECASE,
)
_OBJECT_DESCRIPTORS = re.compile(
    r"(?:(?:a|an|the|red|gray|grey|bright|green|blue|black|white|brown|pink|orange|purple|yellow|tan|dark|light|pale|small|large)\s+){1,3}$",
    re.IGNORECASE,
)
_RELATIVE_CLAUSE = re.compile(r"\s+\b(?:who|that)\b\s+", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(?:enter(?:s|ed|ing)?|leave(?:s|d|ing)?|walk(?:s|ed|ing)?|run(?:s|ning)?|"
    r"crack(?:s|ed|ing)?|cook(?:s|ed|ing)?|mak(?:e|es|ing)|made|cut(?:s|ting)?|"
    r"sit(?:s|ting)?|stand(?:s|ing)?|open(?:s|ed|ing)?|close(?:s|d|ing)?|"
    r"pick(?:s|ed|ing)?|put(?:s|ting)?|eat(?:s|ing)?|drink(?:s|ing)?|"
    r"talk(?:s|ed|ing)?|carry|carries|carried|carrying|hold(?:s|ing)?|push(?:es|ed|ing)?|"
    r"pull(?:s|ed|ing)?|turn(?:s|ed|ing)?|sit(?:s|ting)?|stand(?:s|ing)?)\b",
    re.IGNORECASE,
)
_CARRY_ACTION = re.compile(r"^(?:carry|carries|carried|carrying|hold|holds|holding)$", re.IGNORECASE)
_OBJECTS = (
    "backpack", "rucksack", "handbag", "bag", "suitcase", "hat", "cap",
    "umbrella", "shirt", "jacket", "coat", "dress", "uniform", "shoes",
    "pizza", "cup", "mug", "bottle", "box", "door", "gate", "phone",
    "chair", "table", "vehicle", "car", "bike", "bicycle", "bus", "building",
)
_OBJECT_RE = re.compile(r"\b(" + "|".join(map(re.escape, _OBJECTS)) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryParts:
    """Recognized spans; `original_query` is always preserved verbatim."""

    original_query: str
    person_text: str | None
    object_texts: tuple[str, ...]
    appearance_text: str | None
    action_text: str | None
    carried_object_texts: tuple[str, ...] = ()
    unresolved_carried_object_texts: tuple[str, ...] = ()
    carried_object_mentions: tuple[str, ...] = ()


def _clean_phrase(text: str) -> str:
    text = re.sub(r"^\s*(?:(?:please\s+)?(?:find|search\s+for|look\s+for|locate|show\s+me)\s+)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n.,;:!?-") or ""


def _person_and_appearance(text: str) -> tuple[str | None, str | None]:
    person_match = _PERSON.search(text)
    person_text = person_match.group(0) if person_match else None
    appearance_text = text if person_match and _APPEARANCE.search(text) else None
    return person_text, appearance_text


def _objects(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(m.group(0).lower() for m in _OBJECT_RE.finditer(text)))


def _carried_objects(text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    matches = []
    noun_re = re.compile(r"\b(" + "|".join(
        re.escape(name) for name in sorted(_CARRIED_OBJECT_ALIASES, key=len, reverse=True)
    ) + r")\b", re.IGNORECASE)
    for match in noun_re.finditer(text):
        # Prefer the longest alias at a position (for example, "cell phone" over "phone").
        if any(start <= match.start() < end for start, end, _, _ in matches):
            continue
        matches.append((match.start(), match.end(), _CARRIED_OBJECT_ALIASES[match.group(1).lower()], True))
    for match in _UNSUPPORTED_CARRY_OBJECT_RE.finditer(text):
        phrase = " ".join(" ".join(part for part in (match.group("modifier"), match.group("object")) if part).split())
        phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase, flags=re.IGNORECASE)
        matches.append((match.start(), match.end(), phrase.lower(), False))
    matches.sort()

    supported, unresolved, mentions, accepted = [], [], [], []
    for start, end, value, is_supported in matches:
        context = text[max(0, start - 56):start]
        cue = _CARRIED_TRIGGER.search(context)
        between = text[accepted[-1][1]:start] if accepted else ""
        coordinated = bool(accepted) and re.fullmatch(
            r"\s+and\s+(?:(?:a|an|the|his|her|their|\w+)\s+){0,4}",
            between,
            re.IGNORECASE,
        ) is not None
        if not cue and not coordinated:
            continue
        accepted.append((start, end))
        if is_supported:
            (supported).append(value)
            prefix = _OBJECT_DESCRIPTORS.search(text[max(0, start - 40):start])
            raw = ((prefix.group(0) if prefix else "") + text[start:end]).strip()
            raw = re.sub(r"^(?:a|an|the)\s+", "", raw, flags=re.IGNORECASE)
            mentions.append(" ".join(raw.split()))
        else:
            unresolved.append(value)
            mentions.append(value)
    return tuple(dict.fromkeys(supported)), tuple(dict.fromkeys(unresolved)), tuple(dict.fromkeys(mentions))


def parse_retrieval_query(query: str) -> QueryParts:
    """Extract a few explicit entities, appearance cues, and action phrases.

    The action phrase begins at the first supported action verb, except when
    that verb occurs in a relative clause, where the full relative clause tail
    is retained. Unsupported language leaves semantic spans unset.
    """

    if not isinstance(query, str) or not query.strip():
        return QueryParts(query if isinstance(query, str) else "", None, (), None, None)

    body = _PREFIX.sub("", query, count=1).strip()
    carried_objects, unresolved_carried_objects, carried_mentions = _carried_objects(body)
    relative = _RELATIVE_CLAUSE.search(body)
    if relative:
        subject = _clean_phrase(body[:relative.start()])
        action_tail = _clean_phrase(body[relative.end():])
        action_match = _ACTION.search(action_tail)
        action = action_tail if action_match else None
    else:
        action_match = _ACTION.search(body)
        if action_match:
            subject = _clean_phrase(body[:action_match.start()])
            action = _clean_phrase(body[action_match.start():])
            if _CARRY_ACTION.fullmatch(action_match.group(0)):
                # These verbs are useful as actions but can also be explicit
                # appearance/accessory descriptions in this narrow parser.
                subject = _clean_phrase(body)
                action_tail = body[action_match.end():]
                object_match = _CARRIED_MENTION_RE.search(action_tail) or _UNSUPPORTED_CARRY_OBJECT_RE.search(action_tail)
                if object_match:
                    action_tail = action_tail[:object_match.start()] + action_tail[object_match.end():]
                next_action = _ACTION.search(action_tail)
                action = _clean_phrase(action_tail) if next_action else None
        else:
            subject = _clean_phrase(body)
            action = None

    # A supported carried-object phrase is routed to the detector component,
    # not duplicated as person-crop appearance evidence.
    appearance_subject = _CARRIED_MENTION_RE.sub("", subject)
    appearance_subject = _UNSUPPORTED_CARRY_OBJECT_RE.sub("", appearance_subject)
    appearance_subject = re.sub(r"\s+as\s+they\s*$", "", appearance_subject, flags=re.IGNORECASE)
    appearance_subject = re.sub(r"\b(?:wearing|wears|carrying|carries|carried|holding|holds|with)\s*(?:and|,)?\s*$", "", appearance_subject, flags=re.IGNORECASE)
    appearance_subject = re.sub(r"\s+(?:and|,)\s*$", "", appearance_subject, flags=re.IGNORECASE)
    # For a direct carrying action, keep the appearance span before that action
    # instead of feeding the action tail to the person-crop scorer.
    if action_match and _CARRY_ACTION.fullmatch(action_match.group(0)):
        appearance_subject = _clean_phrase(body[:action_match.start()])
    person_text = _PERSON.search(body)
    person_text = person_text.group(0) if person_text else None
    appearance_text = (
        _clean_phrase(appearance_subject)
        if person_text and _APPEARANCE.search(appearance_subject)
        else None
    )
    return QueryParts(
        original_query=query,
        person_text=person_text,
        object_texts=_objects(body),
        appearance_text=appearance_text,
        action_text=action,
        carried_object_texts=carried_objects,
        unresolved_carried_object_texts=unresolved_carried_objects,
        carried_object_mentions=carried_mentions,
    )


def _valid_scores(scores: Iterable[object]) -> list[float]:
    valid = []
    for score in scores:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append(value)
    return valid


def aggregate_similarity_scores(scores: Iterable[object]) -> float | None:
    """Max-pool valid crop/frame similarities; unavailable evidence stays None."""
    valid = _valid_scores(scores)
    return max(valid) if valid else None


def score_candidate_appearance(
    frames: Iterable[object],
    appearance_text: str | None,
    find_person_crops: Callable[[object], Sequence[object]],
    score_crop: Callable[[object, str], float | None],
) -> float | None:
    """Return the strongest finite crop similarity, or None without evidence."""
    return score_candidate_appearance_evidence(
        frames, appearance_text, find_person_crops, score_crop
    )[0]


def score_candidate_appearance_evidence(
    frames: Iterable[object],
    appearance_text: str | None,
    find_person_crops: Callable[[object], Sequence[object]],
    score_crop: Callable[[object, str], float | None],
) -> tuple[float | None, str]:
    """Return raw max cosine and an explicit evidence availability status."""

    if not isinstance(appearance_text, str) or not appearance_text.strip():
        return None, "not_requested"
    raw_scores = []
    frames_seen = 0
    usable_crops = 0
    detector_successes = 0
    try:
        frame_iterator = iter(frames) if frames is not None else iter(())
    except TypeError:
        frame_iterator = iter(())
    for frame in frame_iterator:
        frames_seen += 1
        try:
            crops = find_person_crops(frame)
        except Exception:
            continue
        detector_successes += 1
        for crop in crops:
            usable_crops += 1
            try:
                raw_scores.append(score_crop(crop, appearance_text))
            except Exception:
                continue
    aggregate = aggregate_similarity_scores(raw_scores)
    if aggregate is not None:
        return aggregate, "scored"
    if not frames_seen:
        return None, "no_frames"
    if not detector_successes:
        return None, "detector_unavailable"
    if not usable_crops:
        return None, "no_usable_person_crop"
    return None, "no_valid_similarity"


def score_candidate_action(
    frames: Iterable[object],
    action_text: str | None,
    score_frame: Callable[[object, str], float | None],
) -> float | None:
    """Mean finite frame/text similarities; this is not motion recognition."""

    if not isinstance(action_text, str) or not action_text.strip():
        return None
    raw_scores = []
    try:
        frame_iterator = iter(frames) if frames is not None else iter(())
    except TypeError:
        frame_iterator = iter(())
    for frame in frame_iterator:
        try:
            raw_scores.append(score_frame(frame, action_text))
        except Exception:
            continue
    scores = _valid_scores(raw_scores)
    return sum(scores) / len(scores) if scores else None


def score_candidate_carried_objects(
    frames: Iterable[object],
    object_texts: Sequence[str],
    detect_objects: Callable[[object, Sequence[str]], Sequence[Mapping]],
) -> tuple[float | None, list[dict], str]:
    """Aggregate positive YOLO category evidence; non-detections stay unknown.

    The returned score is the mean of the strongest matching detector
    confidence per observed requested category. It is neither a calibrated
    probability nor proof that an object is being carried by a person.
    """
    score, detections, status, _ = score_candidate_carried_objects_detailed(
        frames, object_texts, detect_objects
    )
    return score, detections, status


def score_candidate_carried_objects_detailed(
    frames: Iterable[object],
    object_texts: Sequence[str],
    detect_objects: Callable[[object, Sequence[str]], Sequence[Mapping]],
) -> tuple[float | None, list[dict], str, dict[str, str]]:
    """Return positive-only detector evidence and per-category observation states.

    ``observed_negative`` means only that a successful detector pass returned no
    box for that category in the sampled frames. It is never a negative score.
    Unsupported categories and unavailable detection stay scoreless.
    """
    if isinstance(object_texts, str):
        object_texts = (object_texts,)
    try:
        requested = tuple(object_texts or ())
    except (TypeError, ValueError):
        requested = ()
    targets = tuple(dict.fromkeys(
        str(value).strip().lower() for value in requested if str(value).strip()
    ))
    if not targets:
        return None, [], "not_requested", {}
    states = {
        target: ("unknown_unavailable" if target in SUPPORTED_CARRIED_OBJECT_CATEGORIES else "unsupported")
        for target in targets
    }
    supported_targets = tuple(target for target in targets if target in SUPPORTED_CARRIED_OBJECT_CATEGORIES)
    if not supported_targets:
        return None, [], "unsupported_requested_category", states
    try:
        frame_list = list(frames) if frames is not None else []
    except TypeError:
        frame_list = []
    if not frame_list:
        return None, [], "no_frames", states
    found = {}
    successful_frames = 0
    for frame in frame_list:
        try:
            detections = detect_objects(frame, supported_targets)
        except Exception:
            continue
        successful_frames += 1
        for detection in detections or ():
            if not isinstance(detection, Mapping):
                continue
            name = str(detection.get("class_name", "")).strip().lower()
            if name not in supported_targets:
                continue
            scores = _valid_scores([detection.get("confidence")])
            if scores and (name not in found or scores[0] > found[name]["confidence"]):
                found[name] = {**dict(detection), "class_name": name, "confidence": scores[0]}
    if not found:
        if not successful_frames:
            return None, [], "detector_unavailable", states
        for target in supported_targets:
            states[target] = "observed_negative"
        return None, [], "no_matching_detection", states
    for target in supported_targets:
        states[target] = "supported_positive" if target in found else (
            "observed_negative" if successful_frames else "unknown_unavailable"
        )
    return sum(item["confidence"] for item in found.values()) / len(found), list(found.values()), "detected", states


def score_person_object_mentions(
    person_crops: Iterable[object],
    object_mentions: Sequence[str],
    score_crop: Callable[[object, str], float | None],
    score_crop_many: Callable[[object, Sequence[str]], Sequence[float | None]] | None = None,
) -> tuple[float | None, str, list[dict]]:
    """Score all requested object mentions on each individual person crop.

    A candidate score is available only for a single crop with finite scores for
    every distinct mention, preventing evidence from different people being
    combined. The raw cosine remains a similarity, not a calibrated detection.
    """
    if isinstance(object_mentions, str):
        object_mentions = (object_mentions,)
    mentions = tuple(dict.fromkeys(
        str(mention).strip() for mention in (object_mentions or ())
        if isinstance(mention, str) and mention.strip()
    ))
    if not mentions:
        return None, "not_requested", []
    try:
        crops = list(person_crops) if person_crops is not None else []
    except TypeError:
        crops = []
    if not crops:
        return None, "no_person_crop", []
    best = None
    attempted = 0
    for crop_index, crop in enumerate(crops):
        scores = []
        if score_crop_many is not None:
            prompts = [f"person carrying {mention}" for mention in mentions]
            attempted += len(prompts)
            try:
                raw_scores = list(score_crop_many(crop, prompts))
            except Exception:
                raw_scores = []
            if len(raw_scores) == len(mentions):
                for mention, raw in zip(mentions, raw_scores):
                    value = _valid_scores([raw])
                    if value:
                        scores.append((mention, value[0]))
        else:
            for mention in mentions:
                attempted += 1
                try:
                    value = _valid_scores([score_crop(crop, f"person carrying {mention}")])
                except Exception:
                    value = []
                if value:
                    scores.append((mention, value[0]))
        if len(scores) != len(mentions):
            continue
        candidate = (sum(value for _, value in scores) / len(scores), crop_index, scores)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None, "no_valid_person_crop_similarity" if attempted else "no_person_crop", []
    return best[0], "person_crop_text_scored", [
        {"mention": mention, "score": value, "person_crop_index": best[1]}
        for mention, value in best[2]
    ]


def rank_candidates(candidates: Sequence[Mapping], appearance_enabled: bool = True) -> list[dict]:
    """Legacy temporal-primary ordering with appearance as an exact-tie break."""

    indexed = list(enumerate(dict(candidate) for candidate in candidates))

    def key(item):
        index, candidate = item
        temporal = float(candidate["temporal_score"])
        attribute = candidate.get("attribute_score") if appearance_enabled else None
        attribute = float(attribute) if attribute is not None else float("-inf")
        return (-temporal, -attribute, index)

    return [candidate for _, candidate in sorted(indexed, key=key)]


_COMPONENT_FIELDS = {
    "temporal": ("temporal_score", "temporal_rank_score"),
    "appearance": ("appearance_score", "appearance_rank_score"),
    "action": ("action_score", "action_rank_score"),
    "carried_object": ("carried_object_score", "carried_object_rank_score"),
    "person_object": ("person_object_score", "person_object_rank_score"),
}
ABLATION_COMPONENTS = {
    "temporal": ("temporal",),
    "temporal_appearance": ("temporal", "appearance"),
    "temporal_action": ("temporal", "action"),
    "temporal_appearance_action": ("temporal", "appearance", "action"),
    "temporal_appearance_carried_object": ("temporal", "appearance", "carried_object"),
    "temporal_appearance_action_carried_object": ("temporal", "appearance", "action", "carried_object"),
    "temporal_appearance_person_object_carried_object": ("temporal", "appearance", "person_object", "carried_object"),
}


def _normalized_rank_scores(candidates: Sequence[Mapping], component: str) -> dict[int, float]:
    score_field, _ = _COMPONENT_FIELDS[component]
    indexed = []
    for index, candidate in enumerate(candidates):
        raw = candidate.get(score_field)
        if raw is None and component == "appearance":
            raw = candidate.get("attribute_score")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            indexed.append((index, value))
    if not indexed:
        return {}

    ordered = sorted(indexed, key=lambda item: (-item[1], item[0]))
    normalized = {}
    i = 0
    denominator = max(len(ordered) - 1, 1)
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        average_position = (i + j - 1) / 2
        value = 1.0 - average_position / denominator if len(ordered) > 1 else 1.0
        for k in range(i, j):
            normalized[ordered[k][0]] = value
        i = j
    return normalized


def fuse_candidate_ranks(
    candidates: Sequence[Mapping],
    ablation: str = "temporal_appearance_action",
) -> list[dict]:
    """Fuse per-query normalized component ranks with an unweighted mean.

    Raw similarities from different models are not calibrated against each
    other. Each available component is therefore converted to a tied-rank
    percentile within this candidate set. The mean uses only components
    present for that candidate, so absent appearance/action evidence is not
    treated as a zero-valued score. This method is descriptive; no weights are
    learned or tuned on evaluation labels.
    """

    if ablation not in ABLATION_COMPONENTS:
        raise ValueError(f"Unknown ranking ablation: {ablation}")
    source = [dict(candidate) for candidate in candidates]
    components = ABLATION_COMPONENTS[ablation]
    normalized = {
        component: _normalized_rank_scores(source, component)
        for component in components
    }
    results = []
    for index, candidate in enumerate(source):
        evidence = {
            component: values[index]
            for component, values in normalized.items()
            if index in values
        }
        if not evidence:
            fused = None
        else:
            fused = sum(evidence.values()) / len(evidence)
        for component, (_, rank_field) in _COMPONENT_FIELDS.items():
            candidate[rank_field] = normalized.get(component, {}).get(index)
        candidate["fused_score"] = fused
        candidate["fusion_components"] = list(evidence)
        candidate["ranking_method"] = ablation
        results.append((index, candidate))

    return [candidate for _, candidate in sorted(
        results,
        key=lambda item: (
            -(item[1]["fused_score"] if item[1]["fused_score"] is not None else float("-inf")),
            -float(item[1]["temporal_score"]),
            item[0],
        ),
    )]
