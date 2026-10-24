import os
import tempfile
import unittest

from run_on_video.clip_extractor import (
    extract_clip,
    format_timestamp,
    get_video_duration,
    is_within_duration_limit,
    normalize_and_validate,
    normalize_and_validate_scored,
)


EXAMPLE_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "run_on_video",
    "example",
    "RoripwjYFp8_60.0_210.0.mp4",
)


class ClipExtractorTests(unittest.TestCase):
    def test_model_duration_limit_allows_metadata_rounding_only(self):
        self.assertTrue(is_within_duration_limit(150.016533, 150))
        self.assertTrue(is_within_duration_limit(150.1, 150))
        self.assertFalse(is_within_duration_limit(150.1001, 150))
        self.assertFalse(is_within_duration_limit(float("nan"), 150))
        self.assertFalse(is_within_duration_limit(float("inf"), 150))

    def test_timestamp_boundaries(self):
        self.assertEqual(format_timestamp(0), "00:00.00")
        self.assertEqual(format_timestamp(59.999), "01:00.00")
        self.assertEqual(format_timestamp(3600), "01:00:00.00")

    def test_normalization_clamps_video_boundaries_and_marks_invalid_windows(self):
        normalized = normalize_and_validate(
            [[-1, 4, 0.8], [8, 12, 0.7], [10, 11, 0.6], [5, 4, 0.5]],
            video_duration=10,
            padding=2,
        )
        self.assertEqual((normalized[0]["start"], normalized[0]["end"]), (0, 4))
        self.assertEqual((normalized[1]["start"], normalized[1]["end"]), (8, 10))
        self.assertEqual(normalized[0]["padded_start"], 0)
        self.assertEqual(normalized[1]["padded_end"], 10)
        self.assertFalse(normalized[2]["valid"])
        self.assertFalse(normalized[3]["valid"])

    def test_scored_candidates_accept_missing_attribute_and_clamp(self):
        normalized = normalize_and_validate_scored(
            [{"start": -2, "end": 20, "temporal_score": 0.8, "attribute_score": None}],
            video_duration=10,
        )
        self.assertTrue(normalized[0]["valid"])
        self.assertEqual((normalized[0]["start"], normalized[0]["end"]), (0, 10))
        self.assertIsNone(normalized[0]["attribute_score"])

    def test_scored_candidates_preserve_attribute_ranks_and_evidence_status(self):
        normalized = normalize_and_validate_scored(
            [{
                "start": 1,
                "end": 4,
                "temporal_score": 0.8,
                "appearance_score": 0.6,
                "appearance_evidence_status": "scored",
                "action_score": 0.4,
                "carried_object_score": 0.9,
                "carried_object_rank_score": 1.0,
                "carried_object_detections": [{"class_name": "backpack", "confidence": 0.9}],
                "carried_object_evidence_status": "detected",
                "carried_object_evidence_states": {"backpack": "supported_positive"},
                "person_object_score": 0.72,
                "person_object_rank_score": 0.5,
                "person_object_evidence_status": "person_crop_text_scored",
                "person_object_mention_scores": [{"mention": "red backpack", "score": 0.72}],
                "fused_score": 0.75,
                "ranking_method": "temporal_appearance_action_carried_object",
            }],
            video_duration=10,
        )[0]
        self.assertEqual(normalized["appearance_evidence_status"], "scored")
        self.assertEqual(normalized["carried_object_score"], 0.9)
        self.assertEqual(normalized["carried_object_detections"][0]["class_name"], "backpack")
        self.assertEqual(normalized["carried_object_evidence_status"], "detected")
        self.assertEqual(normalized["carried_object_evidence_states"], {"backpack": "supported_positive"})
        self.assertEqual(normalized["person_object_score"], 0.72)
        self.assertEqual(normalized["person_object_evidence_status"], "person_crop_text_scored")
        self.assertEqual(normalized["person_object_mention_scores"][0]["mention"], "red backpack")
        self.assertEqual(normalized["ranking_method"], "temporal_appearance_action_carried_object")

    def test_missing_video_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "missing.mp4")
            with self.assertRaises(Exception):
                get_video_duration(missing)
            ok, _ = extract_clip(missing, 0, 1, temp_dir, 1)
            self.assertFalse(ok)

    @unittest.skipUnless(os.path.isfile(EXAMPLE_VIDEO), "bundled example video is unavailable")
    def test_extracts_real_clip_and_reports_its_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ok, output = extract_clip(EXAMPLE_VIDEO, 2.0, 4.0, temp_dir, 1)
            self.assertTrue(ok, output)
            self.assertTrue(os.path.isfile(output))
            self.assertGreater(os.path.getsize(output), 0)
            duration = get_video_duration(output)
            self.assertGreater(duration, 1.5)
            self.assertLess(duration, 2.5)


if __name__ == "__main__":
    unittest.main()
