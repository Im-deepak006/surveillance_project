import unittest

from standalone_eval.run_track_consistent_feasibility import (
    aggregate_track_scores,
    associate_detections,
    box_iou,
)


class TrackConsistentFeasibilityTests(unittest.TestCase):
    def test_box_iou_for_identical_and_disjoint_boxes(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_iou_association_links_detections_within_gap(self):
        frames = [
            {"timestamp_seconds": 0.0, "person_crops": [{"box_xyxy": [0, 0, 10, 10]}]},
            {"timestamp_seconds": 0.2, "person_crops": [{"box_xyxy": [0, 0, 10, 10]}]},
        ]
        assignments, tracks, stats = associate_detections(frames)
        self.assertEqual(assignments[(0.0, 0)], assignments[(0.2, 0)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(stats["matched_detections"], 1)

    def test_iou_association_breaks_track_after_maximum_gap(self):
        frames = [
            {"timestamp_seconds": 0.0, "person_crops": [{"box_xyxy": [0, 0, 10, 10]}]},
            {"timestamp_seconds": 0.8, "person_crops": [{"box_xyxy": [0, 0, 10, 10]}]},
        ]
        assignments, tracks, _ = associate_detections(frames, max_gap_seconds=0.5)
        self.assertNotEqual(assignments[(0.0, 0)], assignments[(0.8, 0)])
        self.assertEqual(len(tracks), 2)

    def test_candidate_needs_repeated_track_evidence(self):
        candidate = {"appearance_crop_frame_audit": [
            {"timestamp_seconds": 1.0, "person_crops": [{"appearance_similarity": .2}]},
            {"timestamp_seconds": 2.0, "person_crops": [{"appearance_similarity": .8}]},
        ]}
        one_frame = aggregate_track_scores(candidate, {(1.0, 0): 7})
        two_frames = aggregate_track_scores(candidate, {(1.0, 0): 7, (2.0, 0): 7})
        self.assertIsNone(one_frame["score"])
        self.assertEqual(two_frames["score"], .5)
        self.assertEqual(two_frames["eligible_track_count"], 1)


if __name__ == "__main__":
    unittest.main()
