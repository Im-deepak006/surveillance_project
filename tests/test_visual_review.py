import unittest

from standalone_eval.create_visual_review import format_score_track_label, render_interval_timeline


class VisualReviewTests(unittest.TestCase):
    def test_timeline_renders_ground_truth_and_all_prediction_intervals(self):
        svg = render_interval_timeline([[23.5, 35.0]], [
            ("A", (10.0, 18.0, 0.0), "#5286df"),
            ("D", (19.0, 32.0, .57), "#ed9f32"),
            ("T", (16.0, 25.0, .09), "#ab71d0"),
        ], 35)
        self.assertIn('aria-label="Ground truth and proposal intervals"', svg)
        for label in ("Ground truth", "A", "D", "T"):
            self.assertIn(f">{label}</text>", svg)
        self.assertIn("#25a56a", svg)
        self.assertIn("#ab71d0", svg)

    def test_track_label_marks_ids_as_predicted_not_verified(self):
        label = format_score_track_label("appearance", .3047, 357)
        self.assertIn("appearance: 0.305", label)
        self.assertIn("predicted track ID 357", label)
        self.assertNotIn("verified identity", label)

    def test_missing_score_is_explicit_and_track_identity_can_be_marked_verified(self):
        self.assertIn("unavailable in saved artifacts", format_score_track_label("action", None))
        self.assertIn("verified identity 7", format_score_track_label("appearance", .5, 7, True))


if __name__ == "__main__":
    unittest.main()
