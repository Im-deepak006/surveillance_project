# MOT17-01-FRCNN Human Query and Event Annotation

## Scope and separation

This is a blank language-query/event-interval annotation specification for MOT17-01-FRCNN. It is separate from MOT17-04 and from MOT tracking ground truth. Do not copy MOT17-04 queries or event intervals. Do not edit the official MOT tracking labels.

The local video currently found is MOT17-01-FRCNN.mp4. The expected MOT17-01-FRCNN-raw.mp4 filename is not present. Its stream metadata matches the official MOT17-01-FRCNN sequence record (450 frames, 30 fps, 1920x1080), but its byte-for-byte/source provenance has not yet been verified. The official MOTChallenge page places MOT17-01-FRCNN in the test set and provides no ground-truth link for that row. Do not assume MOT17Labels.zip contains this test sequence labels.

## Annotation unit

Create one row per query/event, not per detection. Two human annotators should independently label each query and interval, then adjudicate disagreements. Preserve both original annotations and the adjudicated version. Record the video checksum/version used for annotation.

Required fields:
- query_id: stable sequence-local identifier.
- query_text: natural language description of a visible action/event. State distinguishing clothing/appearance only when visible and unambiguous.
- event_start_frame, event_end_frame_exclusive: 1-based frame interval [start, end). End is exclusive.
- event_start_seconds, event_end_seconds: seconds from the first video frame, using the same half-open interval convention. Derive from the verified video frame rate; do not round the frame labels away.
- target_person_annotation_id: optional human-assigned sequence-local person ID, only when the queried person can be visually and unambiguously followed. Otherwise null. Never substitute a detector or tracker ID.
- person_id_mapping_confidence: high, medium, low, or unavailable, with a short reason.
- event_visibility: clear, partial, or ambiguous.
- annotator_id, annotation_version, notes.

## Labeling rules

1. Mark the visible event onset and offset, not the full time a person remains in the scene. Use a written event definition for each query before labeling.
2. For repeated or interrupted actions, annotate separate intervals unless the query explicitly describes the full sequence.
3. Do not infer backpack/bag ownership from proximity. Annotate an owner relation only when visual evidence supports continuous carrying/handling; otherwise describe the person and object separately or mark the relation ambiguous.
4. Do not force a person GT ID when occlusion, identity ambiguity, or a detector/GT mismatch prevents a reliable mapping.
5. Record uncertainty and adjudication decisions instead of silently resolving ambiguity.
6. Keep temporal language, visible appearance, object category, and person-object relation as distinct fields/evidence.

## Readiness checks before model evaluation

- Confirm the exact video file and checksum.
- Check the contents of any legitimately obtained official labels archive; the MOTChallenge page lists MOT17-01-FRCNN as test and does not link test GT. If no labels are supplied, use the independently created human event/person annotations.
- Verify frame count, frame numbering, image dimensions, and temporal alignment between video and label sequence.
- Complete independent annotation and adjudication for the new queries/events.
- Freeze annotations before evaluating baseline, D, or proposal augmentation.

