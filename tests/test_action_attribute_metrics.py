import unittest

from standalone_eval.action_attribute_metrics import evaluate_ablation, temporal_iou


class ActionAttributeMetricTests(unittest.TestCase):
    def test_temporal_iou_boundaries_and_partial_overlap(self):
        self.assertEqual(temporal_iou([0, 10], [10, 20]), 0.0)
        self.assertEqual(temporal_iou([0, 0], [1, 2]), 0.0)
        self.assertAlmostEqual(temporal_iou([5, 15], [10, 20]), 1 / 3)

    def test_ablation_metrics_use_top_ranked_candidate_and_action_subset(self):
        query = {
            "qid": 7,
            "video": "example.mp4",
            "query": "chef makes pizza",
            "action_text": "makes pizza",
            "appearance_text": None,
            "carried_object_texts": (),
            "ground_truth_intervals": [[10, 20]],
            "candidates": [
                {"start": 0, "end": 5, "temporal_score": 0.9, "action_score": 0.1},
                {"start": 10, "end": 20, "temporal_score": 0.8, "action_score": 0.9},
                {"start": 30, "end": 35, "temporal_score": 0.7, "action_score": 0.5},
            ],
        }
        temporal = evaluate_ablation([query], "temporal")
        action = evaluate_ablation([query], "temporal_action")
        self.assertEqual(temporal["metrics"]["recall_at_1_iou_0_5"], 0.0)
        self.assertEqual(temporal["action_query_metrics"]["query_count"], 1)
        self.assertEqual(action["metrics"]["recall_at_1_iou_0_5"], 1.0)
        self.assertEqual(action["metrics"]["recall_at_1_iou_0_7"], 1.0)
        self.assertEqual(action["metrics"]["mean_iou_at_1"], 1.0)
        self.assertEqual(action["appearance_query_metrics"]["query_count"], 0)
        self.assertEqual(action["combined_query_metrics"]["query_count"], 0)
        self.assertEqual(action["carried_object_query_metrics"]["query_count"], 0)

    def test_appearance_and_combined_query_subsets_are_reported(self):
        query = {
            "query": "person in a red jacket enters",
            "action_text": "enters",
            "appearance_text": "person in a red jacket",
            "carried_object_texts": ("backpack",),
            "ground_truth_intervals": [[0, 10]],
            "candidates": [
                {"start": 0, "end": 10, "temporal_score": 0.8,
                 "appearance_score": 0.7, "action_score": 0.6,
                 "carried_object_score": 0.9}
            ],
        }
        result = evaluate_ablation([query], "temporal_appearance_action_carried_object")
        self.assertEqual(result["appearance_query_metrics"]["query_count"], 1)
        self.assertEqual(result["combined_query_metrics"]["query_count"], 1)
        self.assertEqual(result["carried_object_query_metrics"]["query_count"], 1)
        self.assertIn("carried_object_rank_score", result["queries"][0]["ranked_predictions"][0])


if __name__ == "__main__":
    unittest.main()
