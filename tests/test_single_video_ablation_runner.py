import unittest

from extension.query_logic import fuse_candidate_ranks
from standalone_eval.run_mot17_single_video_ablation import (
    evaluate_rank,
    proposal_oracle,
    summarize_top1,
)


class SingleVideoAblationRunnerTests(unittest.TestCase):
    def test_baseline_and_appearance_use_their_own_fusion_condition(self):
        candidates = [
            {"start": 0, "end": 2, "temporal_score": 0.9, "appearance_score": 0.1},
            {"start": 4, "end": 6, "temporal_score": 0.4, "appearance_score": 0.9},
        ]
        gt = [[4, 6]]

        baseline, baseline_iou = evaluate_rank(candidates, gt, "temporal")
        appearance, appearance_iou = evaluate_rank(candidates, gt, "temporal_appearance")

        self.assertEqual((baseline["start"], baseline["end"]), (0, 2))
        self.assertEqual(baseline_iou, 0)
        # The opposed ranks tie under the project's unweighted mean, so its
        # documented temporal-score tie break keeps the first interval. The
        # method marker verifies the appearance condition was actually used.
        self.assertEqual(appearance["ranking_method"], "temporal_appearance")
        self.assertEqual(appearance["fusion_components"], ["temporal", "appearance"])
        self.assertEqual((appearance["start"], appearance["end"]), (0, 2))
        self.assertEqual(appearance_iou, 0)

    def test_each_ablation_ranks_the_same_proposal_set(self):
        candidates = [
            {"start": 0, "end": 2, "temporal_score": 0.9, "appearance_score": 0.1},
            {"start": 4, "end": 6, "temporal_score": 0.4, "appearance_score": 0.9},
        ]
        proposals = {(item["start"], item["end"]) for item in candidates}
        for ablation in (
            "temporal",
            "temporal_appearance",
            "temporal_appearance_carried_object",
            "temporal_appearance_person_object_carried_object",
        ):
            ranked = fuse_candidate_ranks(candidates, ablation)
            self.assertEqual(
                {(item["start"], item["end"]) for item in ranked}, proposals
            )
            top, _ = evaluate_rank(candidates, [[4, 6]], ablation)
            self.assertIn((top["start"], top["end"]), proposals)
        self.assertEqual({(item["start"], item["end"]) for item in candidates}, proposals)

    def test_metrics_use_condition_specific_eligible_query_denominators(self):
        rows = [
            {"qid": "q1", "conditions": {"A": {"temporal_iou": .6}, "C": {"temporal_iou": .2}}},
            {"qid": "q2", "conditions": {"A": {"temporal_iou": .1}, "C": {"temporal_iou": .8}}},
            {"qid": "q3", "conditions": {"A": {"temporal_iou": .3}, "C": None}},
        ]
        temporal = summarize_top1(rows, "A")
        carried = summarize_top1(rows, "C")
        self.assertEqual(temporal["sample_count"], 3)
        self.assertEqual(temporal["eligible_qids"], ["q1", "q2", "q3"])
        self.assertAlmostEqual(temporal["recall_at_1_iou_0_5"], 1 / 3)
        self.assertAlmostEqual(temporal["recall_at_1_iou_0_3"], 2 / 3)
        self.assertAlmostEqual(temporal["recall_at_1_iou_0_7"], 0)
        self.assertAlmostEqual(temporal["mean_top1_iou"], 1 / 3)
        self.assertEqual(carried["sample_count"], 2)
        self.assertEqual(carried["eligible_qids"], ["q1", "q2"])
        self.assertEqual(carried["recall_at_1_iou_0_5"], .5)

    def test_proposal_oracle_uses_all_proposals_and_reports_exact_hits(self):
        proposals = [
            {"start": 0, "end": 2, "temporal_score": .9},
            {"start": 4, "end": 8, "temporal_score": .8},
            {"start": 8, "end": 10, "temporal_score": .7},
        ]
        oracle = proposal_oracle(proposals, [[4, 8]])
        self.assertEqual(oracle["proposal_count"], 3)
        self.assertEqual(oracle["max_iou"], 1)
        self.assertEqual(oracle["best_proposal_temporal_rank_1_based"], 2)
        self.assertEqual(oracle["proposal_count_at_iou"], {"0.3": 1, "0.5": 1, "0.7": 1})
        self.assertTrue(oracle["any_proposal_reaches_iou_0_5"])


if __name__ == "__main__":
    unittest.main()
