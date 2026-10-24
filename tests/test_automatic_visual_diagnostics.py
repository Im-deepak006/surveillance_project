import math
import unittest

from extension.query_logic import (
    aggregate_similarity_scores,
    score_candidate_appearance_evidence,
    score_candidate_carried_objects_detailed,
)


class AutomaticVisualDiagnosticTests(unittest.TestCase):
    def test_missing_person_crops_remain_unavailable(self):
        score, status = score_candidate_appearance_evidence(
            [object()], "person in a green jacket", lambda frame: [],
            lambda crop, phrase: 0.9,
        )
        self.assertIsNone(score)
        self.assertEqual(status, "no_usable_person_crop")

    def test_unsupported_object_is_not_sent_to_detector_or_scored_negative(self):
        calls = []
        score, detections, status, states = score_candidate_carried_objects_detailed(
            [object()], ["shopping bag"],
            lambda frame, names: calls.append(names) or [],
        )
        self.assertIsNone(score)
        self.assertEqual(detections, [])
        self.assertEqual(status, "unsupported_requested_category")
        self.assertEqual(states, {"shopping bag": "unsupported"})
        self.assertEqual(calls, [])

    def test_multiframe_crop_scores_use_max_and_skip_invalid_values(self):
        self.assertEqual(aggregate_similarity_scores([0.24, None, 0.41, math.nan]), 0.41)
        self.assertIsNone(aggregate_similarity_scores([None, math.nan, "bad"]))


if __name__ == "__main__":
    unittest.main()
