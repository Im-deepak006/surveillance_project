import copy
import unittest

from standalone_eval.run_tracking_consensus_ablation import (
    IoUTrackAssociator,
    aggregate_track_phrase_evidence,
    build_track_candidate,
    match_cached_crops,
    rank_conditions,
    run_bytetrack_frame,
    tracked_detections_from_result,
)


class TrackingConsensusAblationTests(unittest.TestCase):
    def test_bytetrack_uses_default_yaml_and_persistent_video_state(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value
            def cpu(self):
                return self
            def numpy(self):
                return self.value

        class FakeBoxes:
            def __init__(self):
                self.xyxy = FakeTensor([[1, 2, 11, 12]])
                self.conf = FakeTensor([.9])
                self.id = FakeTensor([7])
            def __len__(self):
                return 1

        class FakeResult:
            boxes = FakeBoxes()

        class FakeModel:
            def __init__(self):
                self.calls = []
            def track(self, frame, **kwargs):
                self.calls.append(kwargs)
                return [FakeResult()]

        model = FakeModel()
        detections = run_bytetrack_frame(model, object())
        self.assertEqual(model.calls[0]["persist"], True)
        self.assertEqual(model.calls[0]["tracker"], "bytetrack.yaml")
        self.assertNotIn("conf", model.calls[0])
        self.assertEqual(detections[0]["track_id"], 7)
        self.assertEqual(detections[0]["box_xyxy"], [1, 2, 11, 12])

    def test_bytetrack_detections_without_ids_fail_explicitly(self):
        class FakeTensor:
            def cpu(self):
                return self
            def numpy(self):
                return [[0, 0, 10, 10]]
        class FakeBoxes:
            xyxy = FakeTensor()
            conf = FakeTensor()
            id = None
            def __len__(self):
                return 1
        with self.assertRaisesRegex(RuntimeError, "without track IDs"):
            tracked_detections_from_result(type("Result", (), {"boxes": FakeBoxes()})())

    def test_chronological_processing_and_track_id_persistence(self):
        tracker = IoUTrackAssociator(iou_threshold=.3, max_age_frames=3)
        first = tracker.update(0, 0.0, [{"box_xyxy": [0, 0, 10, 10], "confidence": .9}])
        next_frame = tracker.update(1, 1/30, [{"box_xyxy": [1, 0, 11, 10], "confidence": .8}])
        self.assertEqual(first[0]["track_id"], next_frame[0]["track_id"])
        with self.assertRaises(ValueError):
            tracker.update(1, 2/30, [])

    def test_cached_scores_match_by_timestamp_frame_bbox_not_list_order(self):
        cached = [{"box_xyxy": [0, 0, 10, 10]}, {"box_xyxy": [30, 0, 40, 10]}]
        detections = [
            {"box_xyxy": [30, 0, 40, 10], "track_id": 2},
            {"box_xyxy": [0, 0, 10, 10], "track_id": 1},
        ]
        matched = match_cached_crops(cached, detections, min_iou=.5)
        self.assertEqual(matched["matches"][0]["track_id"], 1)
        self.assertEqual(matched["matches"][1]["track_id"], 2)

    def test_multiple_tracks_keep_phrase_evidence_separate(self):
        observations = [
            {"track_id": 1, "frame_index": 0, "appearance_score": .2, "action_score": .8},
            {"track_id": 1, "frame_index": 1, "appearance_score": .4, "action_score": .6},
            {"track_id": 2, "frame_index": 0, "appearance_score": .7, "action_score": .3},
            {"track_id": 2, "frame_index": 1, "appearance_score": .9, "action_score": .5},
        ]
        result = aggregate_track_phrase_evidence(observations, min_observations=2)
        self.assertEqual(result["appearance"]["best_track_id"], 2)
        self.assertAlmostEqual(result["appearance"]["score"], .8)
        self.assertEqual(result["action"]["best_track_id"], 1)
        self.assertAlmostEqual(result["action"]["score"], .7)
        self.assertEqual(result["appearance"]["eligible_track_count"], 2)
        self.assertEqual(result["action"]["eligible_track_count"], 2)

    def test_missing_or_insufficient_track_evidence_stays_unknown(self):
        result = aggregate_track_phrase_evidence([
            {"track_id": 5, "frame_index": 0, "appearance_score": .9, "action_score": None},
            {"track_id": 5, "frame_index": 1, "appearance_score": None, "action_score": .4},
        ], min_observations=2)
        self.assertIsNone(result["appearance"]["score"])
        self.assertIsNone(result["action"]["score"])
        self.assertEqual(result["appearance"]["status"], "unknown_insufficient_track_evidence")

    def test_phrase_aggregation_uses_median_per_track(self):
        result = aggregate_track_phrase_evidence([
            {"track_id": 3, "frame_index": 0, "appearance_score": .1, "action_score": .3},
            {"track_id": 3, "frame_index": 1, "appearance_score": .9, "action_score": .7},
            {"track_id": 3, "frame_index": 2, "appearance_score": .5, "action_score": .5},
        ], min_observations=2)
        self.assertAlmostEqual(result["appearance"]["score"], .5)
        self.assertAlmostEqual(result["action"]["score"], .5)

    def test_unverified_object_ownership_is_excluded_from_T(self):
        original = {"start": 0, "end": 1, "temporal_score": .5,
                    "appearance_score": .3, "action_score": .2,
                    "person_object_score": .8, "carried_object_score": .9}
        tracked = build_track_candidate(original, [
            {"track_id": 1, "frame_index": 0, "appearance_score": .4, "action_score": .5},
            {"track_id": 1, "frame_index": 1, "appearance_score": .6, "action_score": .7},
        ])
        self.assertIsNone(tracked["person_object_score"])
        self.assertIsNone(tracked["carried_object_score"])
        self.assertEqual(tracked["person_object_evidence_status"], "excluded_unverified_ownership")
        self.assertEqual(tracked["saved_person_object_score"], .8)

    def test_ranking_is_reproducible_and_d_baseline_input_is_unchanged(self):
        original = [
            {"start": 0, "end": 1, "temporal_score": .4, "appearance_score": .2,
             "action_score": .5, "person_object_score": .9, "carried_object_score": .8},
            {"start": 1, "end": 2, "temporal_score": .8, "appearance_score": .7,
             "action_score": .3, "person_object_score": .1, "carried_object_score": .2},
        ]
        before = copy.deepcopy(original)
        tracked = [build_track_candidate(c, [
            {"track_id": i+1, "frame_index": 0, "appearance_score": c["appearance_score"], "action_score": c["action_score"]},
            {"track_id": i+1, "frame_index": 1, "appearance_score": c["appearance_score"], "action_score": c["action_score"]},
        ]) for i, c in enumerate(original)]
        first = rank_conditions(original, tracked)
        second = rank_conditions(original, tracked)
        self.assertEqual(original, before)
        self.assertEqual([(x["start"], x["end"]) for x in first["D"]],
                         [(x["start"], x["end"]) for x in second["D"]])
        self.assertEqual([(x["start"], x["end"]) for x in first["T"]],
                         [(x["start"], x["end"]) for x in second["T"]])
        self.assertTrue(all("person_object" not in x["fusion_components"] and
                            "carried_object" not in x["fusion_components"] for x in first["T"]))


if __name__ == "__main__":
    unittest.main()
