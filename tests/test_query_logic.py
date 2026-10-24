import unittest

from extension.query_logic import (
    fuse_candidate_ranks,
    parse_retrieval_query,
    rank_candidates,
    score_candidate_action,
    score_candidate_appearance,
    score_candidate_appearance_evidence,
    score_candidate_carried_objects,
    score_candidate_carried_objects_detailed,
    score_person_object_mentions,
    sample_interval_timestamps,
)


class QueryParsingTests(unittest.TestCase):
    def test_single_frame_sampling_uses_proposal_midpoint(self):
        self.assertEqual(sample_interval_timestamps(10.0, 20.0, 1), [15.0])

    def test_multiple_frame_sampling_includes_both_interval_boundaries(self):
        self.assertEqual(sample_interval_timestamps(10.0, 20.0, 3), [10.0, 15.0, 20.0])

    def test_frame_sampling_rejects_invalid_intervals_and_counts(self):
        for interval, count in [((1, 1), 1), ((0, float("inf")), 1), ((0, 1), 0)]:
            with self.subTest(interval=interval, count=count):
                with self.assertRaises(ValueError):
                    sample_interval_timestamps(interval[0], interval[1], count)

    def test_splits_supported_appearance_and_relative_action(self):
        parts = parse_retrieval_query(
            "Find the person wearing a blue shirt and carrying a backpack "
            "who enters the building."
        )
        self.assertEqual(
            parts.appearance_text,
            "person wearing a blue shirt",
        )
        self.assertEqual(parts.carried_object_texts, ("backpack",))
        self.assertEqual(parts.action_text, "enters the building")
        self.assertEqual(parts.original_query, "Find the person wearing a blue shirt and carrying a backpack who enters the building.")

    def test_appearance_without_action(self):
        parts = parse_retrieval_query("Find the person carrying a backpack.")
        self.assertIsNone(parts.appearance_text)
        self.assertIsNone(parts.action_text)
        self.assertEqual(parts.carried_object_texts, ("backpack",))

    def test_wearing_a_backpack_is_object_evidence_not_clothing_text(self):
        parts = parse_retrieval_query("Find the man wearing a backpack.")
        self.assertIsNone(parts.appearance_text)
        self.assertEqual(parts.carried_object_texts, ("backpack",))

    def test_clothing_and_carried_object_are_separated(self):
        parts = parse_retrieval_query("Find a woman in a red jacket with a black handbag.")
        self.assertEqual(parts.appearance_text, "woman in a red jacket")
        self.assertEqual(parts.carried_object_texts, ("handbag",))

    def test_unrecognized_bag_phrase_does_not_invent_supported_category(self):
        parts = parse_retrieval_query("Find a person holding a shopping tote.")
        self.assertEqual(parts.carried_object_texts, ())
        self.assertEqual(parts.unresolved_carried_object_texts, ())

    def test_five_mot17_pilot_queries_are_decomposed_without_losing_cues(self):
        cases = [
            (
                "A person in a dark jacket and tan trousers wears a red backpack as they walk away from the camera.",
                "person in a dark jacket and tan trousers", "walk away from the camera", ("backpack",), (), ("red backpack",),
            ),
            (
                "A person in a green jacket and blue trousers wears a gray backpack and a bright yellow shoulder bag as they walk away from the camera.",
                "person in a green jacket and blue trousers", "walk away from the camera", ("backpack", "handbag"), (), ("gray backpack", "bright yellow shoulder bag"),
            ),
            (
                "A person in a white shirt and dark trousers carries a bright green shopping bag while walking across the lower-left foreground.",
                "person in a white shirt and dark trousers", "while walking across the lower-left foreground", (), ("bright green shopping bag",), ("bright green shopping bag",),
            ),
            (
                "A person in a white shirt and dark trousers walks toward the camera along the left side.",
                "person in a white shirt and dark trousers", "walks toward the camera along the left side", (), (), (),
            ),
            (
                "A person in a pale pink shirt and dark trousers walks beside the white-shirted person toward the camera.",
                "person in a pale pink shirt and dark trousers", "walks beside the white-shirted person toward the camera", (), (), (),
            ),
        ]
        for query, appearance, action, objects, unresolved, mentions in cases:
            with self.subTest(query=query):
                parts = parse_retrieval_query(query)
                self.assertEqual(parts.appearance_text, appearance)
                self.assertEqual(parts.action_text, action)
                self.assertEqual(parts.carried_object_texts, objects)
                self.assertEqual(parts.unresolved_carried_object_texts, unresolved)
                self.assertEqual(parts.carried_object_mentions, mentions)

    def test_carried_object_phrase_is_not_duplicated_into_appearance_or_action(self):
        parts = parse_retrieval_query(
            "A person wearing a red shirt carries a red backpack while walking."
        )
        self.assertEqual(parts.appearance_text, "person wearing a red shirt")
        self.assertEqual(parts.action_text, "while walking")
        self.assertEqual(parts.carried_object_texts, ("backpack",))
        self.assertEqual(parts.carried_object_mentions, ("red backpack",))

    def test_multiple_requested_objects_are_preserved_and_normalized(self):
        parts = parse_retrieval_query(
            "A person with a backpack and a blue handbag walks past the gate."
        )
        self.assertEqual(parts.carried_object_texts, ("backpack", "handbag"))
        self.assertEqual(parts.carried_object_mentions, ("backpack", "blue handbag"))

    def test_unknown_shopping_bag_is_preserved_but_not_mapped_to_yolo_class(self):
        parts = parse_retrieval_query("A person carries a red shopping bag.")
        self.assertEqual(parts.carried_object_texts, ())
        self.assertEqual(parts.unresolved_carried_object_texts, ("red shopping bag",))
        self.assertEqual(parts.carried_object_mentions, ("red shopping bag",))
        self.assertIsNone(parts.action_text)

    def test_plural_and_coco_supported_carried_objects_are_normalized(self):
        cases = {
            "A man carrying suitcases walks through the door": ("suitcase",),
            "A woman holding a green bottle": ("bottle",),
            "A person with a mobile phone": ("cell phone",),
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(parse_retrieval_query(query).carried_object_texts, expected)

    def test_action_without_appearance(self):
        parts = parse_retrieval_query("Find the person who enters the building.")
        self.assertIsNone(parts.appearance_text)
        self.assertEqual(parts.action_text, "enters the building")

    def test_ambiguous_order_falls_back_without_appearance(self):
        parts = parse_retrieval_query("Find the person entering the building in blue.")
        self.assertIsNone(parts.appearance_text)

    def test_unsplit_action_keeps_subject_appearance_and_action_separate(self):
        parts = parse_retrieval_query(
            "Find the woman in a black dress cracking walnuts."
        )
        self.assertEqual(parts.appearance_text, "woman in a black dress")
        self.assertEqual(parts.action_text, "cracking walnuts")

    def test_empty_or_non_string_query(self):
        for query in ("", "  ", None):
            with self.subTest(query=query):
                parts = parse_retrieval_query(query)
                self.assertIsNone(parts.appearance_text)
                self.assertIsNone(parts.action_text)

    def test_unsupported_query_falls_back_to_temporal_only(self):
        for query in (
            "an animal is painted turquoise",
            "a person in patterned coveralls beside the fence",
        ):
            with self.subTest(query=query):
                parts = parse_retrieval_query(query)
                self.assertIsNone(parts.appearance_text)

    def test_supported_clothing_text_and_action_are_separated(self):
        parts = parse_retrieval_query("Someone in a blue shirt who walks near the gate")
        self.assertEqual(parts.appearance_text, "Someone in a blue shirt")
        self.assertEqual(parts.action_text, "walks near the gate")



class CandidateRankingTests(unittest.TestCase):
    def test_temporal_score_remains_primary(self):
        ranked = rank_candidates(
            [
                {"id": "temporal", "temporal_score": 0.9, "attribute_score": -0.5},
                {"id": "appearance", "temporal_score": 0.8, "attribute_score": 0.9},
            ]
        )
        self.assertEqual([c["id"] for c in ranked], ["temporal", "appearance"])

    def test_appearance_breaks_temporal_ties(self):
        ranked = rank_candidates(
            [
                {"id": "first", "temporal_score": 0.7, "attribute_score": 0.1},
                {"id": "second", "temporal_score": 0.7, "attribute_score": 0.8},
            ]
        )
        self.assertEqual([c["id"] for c in ranked], ["second", "first"])

    def test_equal_scores_keep_input_order(self):
        ranked = rank_candidates(
            [
                {"id": "first", "temporal_score": 0.7, "attribute_score": 0.4},
                {"id": "second", "temporal_score": 0.7, "attribute_score": 0.4},
            ]
        )
        self.assertEqual([c["id"] for c in ranked], ["first", "second"])

    def test_empty_detections_produce_no_attribute_evidence(self):
        score = score_candidate_appearance(
            frames=["frame"],
            appearance_text="person wearing a blue shirt",
            find_person_crops=lambda frame: [],
            score_crop=lambda crop, text: self.fail("no crop should be scored"),
        )
        self.assertIsNone(score)

    def test_no_appearance_skips_crop_scoring(self):
        score = score_candidate_appearance(
            frames=["frame"],
            appearance_text=None,
            find_person_crops=lambda frame: self.fail("frames should not be scanned"),
            score_crop=lambda crop, text: self.fail("no crop should be scored"),
        )
        self.assertIsNone(score)

    def test_valid_crops_are_scored_and_best_similarity_is_kept(self):
        seen = []
        score = score_candidate_appearance(
            frames=["frame-1", "frame-2"],
            appearance_text="person wearing a blue shirt",
            find_person_crops=lambda frame: [frame + "-crop", frame + "-crop-2"],
            score_crop=lambda crop, text: seen.append((crop, text)) or len(seen) / 10,
        )
        self.assertEqual(len(seen), 4)
        self.assertEqual(score, 0.4)

    def test_invalid_crop_score_is_ignored_as_missing_evidence(self):
        score = score_candidate_appearance(
            frames=["frame"],
            appearance_text="person wearing a blue shirt",
            find_person_crops=lambda frame: [object()],
            score_crop=lambda crop, text: "not-a-number",
        )
        self.assertIsNone(score)

    def test_invalid_appearance_and_action_inputs_return_no_evidence(self):
        self.assertIsNone(score_candidate_appearance(None, 123, lambda *_: [], lambda *_: 0.1))
        self.assertIsNone(score_candidate_action([1], 123, lambda *_: 0.1))

    def test_appearance_evidence_reports_missing_person_crops(self):
        score, status = score_candidate_appearance_evidence(
            ["frame"],
            "person in a blue shirt",
            find_person_crops=lambda frame: [],
            score_crop=lambda crop, text: 0.5,
        )
        self.assertIsNone(score)
        self.assertEqual(status, "no_usable_person_crop")

    def test_appearance_evidence_reports_detector_failure_and_no_frames(self):
        _, detector_status = score_candidate_appearance_evidence(
            ["frame"], "person in a blue shirt",
            find_person_crops=lambda frame: (_ for _ in ()).throw(RuntimeError("detector")),
            score_crop=lambda crop, text: 0.5,
        )
        _, empty_status = score_candidate_appearance_evidence(
            [], "person in a blue shirt", lambda frame: [], lambda crop, text: 0.5
        )
        self.assertEqual(detector_status, "detector_unavailable")
        self.assertEqual(empty_status, "no_frames")

    def test_action_score_aggregates_finite_frame_similarities(self):
        seen = []
        score = score_candidate_action(
            frames=[1, 2, 3],
            action_text="makes pizza and cuts it up",
            score_frame=lambda frame, action: seen.append(frame) or {1: 0.2, 2: None, 3: 0.6}[frame],
        )
        self.assertEqual(seen, [1, 2, 3])
        self.assertAlmostEqual(score, 0.4)

    def test_missing_action_or_frames_has_no_action_evidence(self):
        self.assertIsNone(score_candidate_action([], "walks", lambda frame, text: 1.0))
        self.assertIsNone(score_candidate_action(None, "walks", lambda frame, text: 1.0))
        self.assertIsNone(score_candidate_action([1], None, lambda frame, text: 1.0))

    def test_carried_object_score_uses_detected_confidence_only(self):
        score, detections, status = score_candidate_carried_objects(
            ["frame-1", "frame-2"],
            ["backpack", "handbag"],
            lambda frame, targets: (
                [{"class_name": "backpack", "confidence": 0.8}]
                if frame == "frame-1" else []
            ),
        )
        self.assertEqual(score, 0.8)
        self.assertEqual(detections[0]["class_name"], "backpack")
        self.assertEqual(status, "detected")

    def test_absent_carried_object_detections_are_unknown_not_zero(self):
        score, detections, status = score_candidate_carried_objects(
            ["frame"], ["backpack"], lambda frame, targets: []
        )
        self.assertIsNone(score)
        self.assertEqual(detections, [])
        self.assertEqual(status, "no_matching_detection")

    def test_unavailable_object_frames_and_unrequested_object_are_explicit(self):
        self.assertEqual(
            score_candidate_carried_objects([], ["backpack"], lambda *_: [])[2],
            "no_frames",
        )
        self.assertEqual(
            score_candidate_carried_objects(["frame"], [], lambda *_: [])[2],
            "not_requested",
        )
        self.assertEqual(
            score_candidate_carried_objects(None, "backpack", lambda *_: [])[2],
            "no_frames",
        )

    def test_all_object_detector_errors_are_not_misreported_as_absence(self):
        score, detections, status = score_candidate_carried_objects(
            ["frame"], ["backpack"],
            lambda *_: (_ for _ in ()).throw(RuntimeError("detector")),
        )
        self.assertIsNone(score)
        self.assertEqual(detections, [])
        self.assertEqual(status, "detector_unavailable")

    def test_object_states_separate_positive_observed_non_detection_and_unknown(self):
        score, _, status, states = score_candidate_carried_objects_detailed(
            ["frame"], ["backpack", "handbag"],
            lambda *_: [{"class_name": "backpack", "confidence": .7}],
        )
        self.assertEqual(status, "detected")
        self.assertEqual(score, .7)
        self.assertEqual(states, {"backpack": "supported_positive", "handbag": "observed_negative"})

        _, _, _, unavailable = score_candidate_carried_objects_detailed(
            ["frame"], ["backpack"],
            lambda *_: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        self.assertEqual(unavailable["backpack"], "unknown_unavailable")

    def test_unsupported_category_never_calls_detector_or_becomes_negative(self):
        score, detections, status, states = score_candidate_carried_objects_detailed(
            ["frame"], ["shopping bag"],
            lambda *_: self.fail("unsupported category must not reach YOLO"),
        )
        self.assertIsNone(score)
        self.assertEqual(detections, [])
        self.assertEqual(status, "unsupported_requested_category")
        self.assertEqual(states, {"shopping bag": "unsupported"})

        score, _, status, states = score_candidate_carried_objects_detailed(
            ["frame"], ["backpack", "shopping bag"],
            lambda frame, supported: self.assertEqual(supported, ("backpack",)) or [],
        )
        self.assertIsNone(score)
        self.assertEqual(status, "no_matching_detection")
        self.assertEqual(states, {"backpack": "observed_negative", "shopping bag": "unsupported"})

    def test_person_object_mentions_must_score_on_one_person_crop(self):
        score, status, mention_scores = score_person_object_mentions(
            ["backpack-person", "bag-person"], ["gray backpack", "yellow shoulder bag"],
            lambda crop, text: .9 if "gray backpack" in text and crop == "backpack-person"
            else .8 if "yellow shoulder bag" in text and crop == "backpack-person"
            else .99 if "yellow shoulder bag" in text and crop == "bag-person"
            else None,
        )
        self.assertEqual(status, "person_crop_text_scored")
        self.assertAlmostEqual(score, .85)
        self.assertEqual({item["person_crop_index"] for item in mention_scores}, {0})

        score, status, mention_scores = score_person_object_mentions(
            ["backpack-person", "bag-person"], ["gray backpack", "yellow shoulder bag"],
            lambda crop, text: .9 if "gray backpack" in text and crop == "backpack-person"
            else .9 if "yellow shoulder bag" in text and crop == "bag-person"
            else None,
        )
        self.assertIsNone(score)
        self.assertEqual(status, "no_valid_person_crop_similarity")
        self.assertEqual(mention_scores, [])

    def test_person_object_mentions_support_batched_phrase_scores_per_crop(self):
        calls = []

        def score_many(crop, prompts):
            calls.append((crop, prompts))
            return [.8, .6]

        score, status, details = score_person_object_mentions(
            ["crop-a"],
            ["gray backpack", "yellow shoulder bag"],
            lambda *_: self.fail("batch scorer should be used"),
            score_crop_many=score_many,
        )
        self.assertEqual(score, .7)
        self.assertEqual(status, "person_crop_text_scored")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ["person carrying gray backpack", "person carrying yellow shoulder bag"])
        self.assertEqual([item["score"] for item in details], [.8, .6])

    def test_action_query_decomposition_keeps_subject_and_object(self):
        query = "Chef makes pizza and cuts it up."
        parts = parse_retrieval_query(query)
        self.assertEqual(parts.original_query, query)
        self.assertEqual(parts.person_text, "Chef")
        self.assertEqual(parts.object_texts, ("pizza",))
        self.assertEqual(parts.action_text, "makes pizza and cuts it up")
        self.assertIsNone(parts.appearance_text)

    def test_missing_appearance_scores_leave_temporal_order(self):
        ranked = rank_candidates(
            [
                {"id": "first", "temporal_score": 0.7, "attribute_score": None},
                {"id": "second", "temporal_score": 0.7, "attribute_score": None},
            ]
        )
        self.assertEqual([c["id"] for c in ranked], ["first", "second"])


class RankFusionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "temporal", "temporal_score": 0.9, "appearance_score": 0.1, "action_score": 0.1},
            {"id": "balanced", "temporal_score": 0.8, "appearance_score": 0.9, "action_score": 0.8},
            {"id": "semantic", "temporal_score": 0.7, "appearance_score": 0.8, "action_score": 0.9},
        ]

    def test_temporal_ablation_matches_baseline_and_records_normalized_scores(self):
        fused = fuse_candidate_ranks(self.candidates, "temporal")
        self.assertEqual([c["id"] for c in fused], ["temporal", "balanced", "semantic"])
        self.assertEqual(fused[0]["temporal_rank_score"], 1.0)
        self.assertIsNone(fused[0]["appearance_rank_score"])
        self.assertEqual(fused[0]["fusion_components"], ["temporal"])

    def test_component_rank_fusion_changes_order_and_keeps_raw_scores(self):
        fused = fuse_candidate_ranks(self.candidates, "temporal_appearance_action")
        self.assertEqual([c["id"] for c in fused], ["balanced", "semantic", "temporal"])
        balanced = next(c for c in fused if c["id"] == "balanced")
        self.assertEqual(balanced["appearance_score"], 0.9)
        self.assertEqual(balanced["action_score"], 0.8)
        self.assertEqual(balanced["fusion_components"], ["temporal", "appearance", "action"])

    def test_missing_evidence_is_omitted_not_zero_filled(self):
        candidates = [
            {"id": "no-evidence", "temporal_score": 0.9, "appearance_score": None, "action_score": None},
            {"id": "has-evidence", "temporal_score": 0.8, "appearance_score": 0.5, "action_score": 0.2},
        ]
        fused = fuse_candidate_ranks(candidates, "temporal_appearance_action")
        missing = next(c for c in fused if c["id"] == "no-evidence")
        self.assertEqual(missing["fusion_components"], ["temporal"])
        self.assertEqual(missing["fused_score"], missing["temporal_rank_score"])

    def test_carried_object_ablation_adds_only_observed_object_evidence(self):
        candidates = [
            {"id": "detected", "temporal_score": 0.7, "carried_object_score": 0.9},
            {"id": "unknown", "temporal_score": 0.8, "carried_object_score": None},
        ]
        fused = fuse_candidate_ranks(candidates, "temporal_appearance_action_carried_object")
        detected = next(c for c in fused if c["id"] == "detected")
        unknown = next(c for c in fused if c["id"] == "unknown")
        self.assertIn("carried_object", detected["fusion_components"])
        self.assertNotIn("carried_object", unknown["fusion_components"])


if __name__ == "__main__":
    unittest.main()
