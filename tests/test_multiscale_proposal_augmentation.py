import unittest

from extension.query_logic import fuse_candidate_ranks
from standalone_eval.run_multiscale_proposal_augmentation import (
    D_FUSION,
    generate_augmented_candidates,
    merge_fresh_evidence,
    render_interval_svg,
)


class MultiScaleProposalAugmentationTests(unittest.TestCase):
    def test_generates_the_four_predeclared_transforms(self):
        original = [{"start": 4.0, "end": 8.0, "temporal_score": 0.7}]
        kept, generated, counts = generate_augmented_candidates(original, 20.0)

        self.assertEqual(len(kept), 1)
        self.assertEqual(
            {(row["start"], row["end"]) for row in generated},
            {(5.0, 7.0), (3.0, 9.0), (2.0, 6.0), (6.0, 10.0)},
        )
        self.assertEqual(counts["generated_unique"], 4)
        self.assertEqual(
            {row["augmentation_transform"] for row in generated},
            {"duration_scale_0.5x", "duration_scale_1.5x", "center_shift_-2s", "center_shift_+2s"},
        )

    def test_generated_intervals_are_clipped_to_video_bounds(self):
        _, generated, _ = generate_augmented_candidates(
            [{"start": 0.0, "end": 4.0, "temporal_score": 0.8}], 10.0
        )
        self.assertIn((0.0, 5.0), {(row["start"], row["end"]) for row in generated})
        self.assertIn((0.0, 2.0), {(row["start"], row["end"]) for row in generated})
        self.assertTrue(all(0.0 <= row["start"] < row["end"] <= 10.0 for row in generated))

    def test_non_positive_intervals_after_clipping_are_discarded(self):
        _, generated, counts = generate_augmented_candidates(
            [{"start": 0.0, "end": 1.0, "temporal_score": 0.8}], 10.0
        )
        self.assertGreaterEqual(counts["invalid_discarded"], 1)
        self.assertTrue(all(row["end"] > row["start"] for row in generated))

    def test_suppresses_collisions_with_original_and_deduplicates_generated(self):
        parents = [
            {"start": 3.0, "end": 7.0, "temporal_score": 0.7},
            {"start": 0.0, "end": 6.0, "temporal_score": 0.9},
            {"start": 2.0, "end": 8.0, "temporal_score": 0.5},
        ]
        _, generated, counts = generate_augmented_candidates(parents, 20.0)
        hits = [row for row in generated if (row["start"], row["end"]) == (2.0, 8.0)]

        # [2, 8] is parent 0 scaled by 1.5 and parent 1 shifted by +2;
        # parent 2's exact original proposal wins over any generated duplicate.
        self.assertEqual(hits, [])
        self.assertGreater(counts["duplicate_or_original_suppressed"], 0)

        _, generated_without_original_collision, _ = generate_augmented_candidates(parents[:2], 20.0)
        hits = [row for row in generated_without_original_collision
                if (row["start"], row["end"]) == (2.0, 8.0)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(len(hits[0]["duplicate_sources"]), 2)
        self.assertEqual(hits[0]["temporal_score"], 0.7)

    def test_original_candidates_and_scores_are_preserved_exactly(self):
        originals = [
            {"start": 1.2345, "end": 5.6789, "temporal_score": 0.83, "appearance_score": 0.41},
            {"start": 7.0, "end": 9.0, "temporal_score": 0.22, "person_object_score": None},
        ]
        kept, _, _ = generate_augmented_candidates(originals, 10.0)
        self.assertEqual(kept, originals)

    def test_parent_temporal_confidence_is_inherited_and_labeled(self):
        _, generated, _ = generate_augmented_candidates(
            [{"start": 4.0, "end": 8.0, "temporal_score": 0.731}], 20.0
        )
        self.assertTrue(generated)
        self.assertTrue(all(row["temporal_score"] == 0.731 for row in generated))
        self.assertTrue(all(row["parent_temporal_score"] == 0.731 for row in generated))
        self.assertTrue(all(
            row["temporal_score_source"] == "inherited_from_parent_proposal"
            for row in generated
        ))
        self.assertTrue(all("appearance_score" not in row for row in generated))

    def test_missing_semantic_evidence_stays_missing_in_d_fusion(self):
        _, generated, _ = generate_augmented_candidates(
            [{"start": 4.0, "end": 8.0, "temporal_score": 0.9}], 20.0
        )
        candidate = merge_fresh_evidence(generated[0], {})
        ranked = fuse_candidate_ranks([candidate], D_FUSION)

        self.assertIsNone(candidate["appearance_score"])
        self.assertIsNone(candidate["person_object_score"])
        self.assertIsNone(candidate["carried_object_score"])
        self.assertEqual(candidate["appearance_evidence_status"], "missing_from_scorer_output")
        self.assertEqual(ranked[0]["fusion_components"], ["temporal"])
        self.assertEqual(ranked[0]["temporal_score_source"], "inherited_from_parent_proposal")

    def test_interval_visual_labels_conditions_and_temporal_iou_context(self):
        def condition(start, end, iou):
            return {"top1": {"start": start, "end": end, "temporal_iou": iou}}

        coverage = {"best_interval": [2.0, 7.0], "max_iou": 0.8}
        svg = render_interval_svg(
            "q1", 20.0, [[1.0, 8.0]],
            {"A": condition(0, 5, 0.4), "D": condition(2, 7, 0.7),
             "augmented_D": condition(3, 8, 0.8)},
            coverage, {"best_interval": [3.0, 8.0], "max_iou": 0.9},
        )
        for label in ("q1 interval comparison", "Ground truth", "A: temporal-only",
                      "D: saved proposals", "D: augmented", "Oracle: originals",
                      "Oracle: augmented", "identity and object ownership are not evaluated"):
            self.assertIn(label, svg)
        self.assertIn("0.00–5.00 s", svg)


if __name__ == "__main__":
    unittest.main()
