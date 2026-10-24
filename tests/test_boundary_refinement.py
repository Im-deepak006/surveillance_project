import unittest

from standalone_eval.boundary_refinement import consensus_boundary_shift, refine_interval


class BoundaryConsensusTests(unittest.TestCase):
    def test_start_and_end_directions(self):
        # Both cues prefer outside: start extends earlier; end extends later.
        self.assertEqual(consensus_boundary_shift("start", .2, .8, .3, .9), (-.5, "moved_earlier"))
        self.assertEqual(consensus_boundary_shift("end", .2, .8, .3, .9), (.5, "moved_later"))
        # Both cues prefer inside: start later; end earlier.
        self.assertEqual(consensus_boundary_shift("start", .8, .2, .9, .3), (.5, "moved_later"))
        self.assertEqual(consensus_boundary_shift("end", .8, .2, .9, .3), (-.5, "moved_earlier"))

    def test_tie_in_either_cue_retains_boundary(self):
        self.assertEqual(consensus_boundary_shift("start", .4, .4, .3, .9), (0.0, "unchanged_tie"))
        self.assertEqual(consensus_boundary_shift("end", .2, .8, .6, .6), (0.0, "unchanged_tie"))

    def test_missing_or_nonfinite_cue_rejects(self):
        self.assertEqual(consensus_boundary_shift("start", None, .4, .3, .5), (0.0, "rejected_missing_evidence"))
        self.assertEqual(consensus_boundary_shift("end", .4, .3, float("nan"), .5),
                         (0.0, "rejected_missing_evidence"))

    def test_conflicting_cues_reject(self):
        self.assertEqual(consensus_boundary_shift("start", .8, .2, .1, .9),
                         (0.0, "rejected_conflicting_evidence"))

    def test_video_limit_clipping(self):
        self.assertEqual(refine_interval(.2, 34.8, -.5, .5, 35.0), (0.0, 35.0, "refined"))

    def test_invalid_interval_restores_original(self):
        self.assertEqual(refine_interval(4.0, 4.3, .5, -.5, 35.0),
                         (4.0, 4.3, "invalid_refinement_fallback"))
        self.assertEqual(refine_interval(8.0, 7.0, .0, .0, 35.0),
                         (8.0, 7.0, "invalid_original_fallback"))


if __name__ == "__main__":
    unittest.main()
