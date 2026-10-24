import unittest

from extension.query_logic import rank_candidates
from run_on_video.clip_extractor import normalize_and_validate_scored


class ClipRankingContractTests(unittest.TestCase):
    def test_temporal_only_candidates_normalize_without_attribute_score(self):
        candidates = rank_candidates(
            [
                {"start": 4, "end": 8, "temporal_score": 0.8, "attribute_score": None},
                {"start": 0, "end": 3, "temporal_score": 0.9, "attribute_score": None},
            ],
            appearance_enabled=False,
        )
        normalized = normalize_and_validate_scored(candidates, video_duration=10)
        self.assertEqual([item["score"] for item in normalized], [0.9, 0.8])
        self.assertEqual([item["attribute_score"] for item in normalized], [None, None])
        self.assertTrue(all(item["valid"] for item in normalized))

    def test_attribute_tie_break_survives_normalization(self):
        candidates = rank_candidates(
            [
                {"id": "weak", "start": 4, "end": 8, "temporal_score": 0.8, "attribute_score": -0.1},
                {"id": "strong", "start": 0, "end": 3, "temporal_score": 0.8, "attribute_score": 0.2},
            ]
        )
        normalized = normalize_and_validate_scored(candidates, video_duration=10)
        self.assertEqual([item["attribute_score"] for item in normalized], [0.2, -0.1])
        self.assertEqual([item["score"] for item in normalized], [0.8, 0.8])


if __name__ == "__main__":
    unittest.main()
