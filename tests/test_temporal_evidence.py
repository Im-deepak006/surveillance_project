import unittest

from extension.temporal_evidence import (
    aggregate_frame_appearance,
    aggregate_repeated_object_evidence,
)


class TemporalEvidenceAggregationTests(unittest.TestCase):
    def test_appearance_uses_median_of_frame_maxima_not_global_max(self):
        result = aggregate_frame_appearance([[.31, .44], [.35], [.92, .20]])
        self.assertEqual(result["frame_maxima"], [.44, .35, .92])
        self.assertAlmostEqual(result["score"], .44)
        self.assertEqual(result["status"], "multi_frame_consensus")

    def test_missing_crop_frames_are_unknown_and_not_zero_filled(self):
        result = aggregate_frame_appearance([[], [None, float("nan")], [.6]])
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "unknown_insufficient_frame_evidence")
        self.assertEqual(result["observed_frame_count"], 1)

    def test_object_category_requires_repeated_positive_detections(self):
        result = aggregate_repeated_object_evidence(
            [[{"class_name": "backpack", "confidence": .8}], [],
             [{"class_name": "backpack", "confidence": .6}]],
            ["backpack"], ["backpack"], min_positive_frames=2,
        )
        self.assertAlmostEqual(result["score"], .7)
        self.assertEqual(result["states"]["backpack"], "supported_positive_repeated")

    def test_missing_or_single_object_detection_stays_unknown(self):
        for detections in ([[], [], []], [[{"class_name": "backpack", "confidence": .9}], [], []]):
            result = aggregate_repeated_object_evidence(
                detections, ["backpack"], ["backpack"], min_positive_frames=2,
            )
            self.assertIsNone(result["score"])
            self.assertEqual(result["states"]["backpack"], "unknown_insufficient_repeat_positive")

    def test_unsupported_object_does_not_become_negative_or_positive(self):
        result = aggregate_repeated_object_evidence(
            [[{"class_name": "backpack", "confidence": .9}] for _ in range(3)],
            ["shopping bag"], ["backpack"],
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["states"]["shopping bag"], "unsupported")

    def test_unrequested_object_returns_not_requested(self):
        result = aggregate_repeated_object_evidence([[], [], []], [], ["backpack"])
        self.assertEqual(result["status"], "not_requested")
        self.assertEqual(result["states"], {})


if __name__ == "__main__":
    unittest.main()
