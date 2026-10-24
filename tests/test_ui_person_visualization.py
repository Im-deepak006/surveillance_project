import unittest

import numpy as np

from extension.ui_person_visualization import (
    build_person_crop_gallery_items,
    crop_person_detections,
    draw_person_boxes,
    midpoint_timestamp,
    person_detection_label,
    person_detections_from_result,
    representative_frame_timestamps,
)


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class UIPersonVisualizationTests(unittest.TestCase):
    def test_midpoint_of_selected_interval(self):
        self.assertEqual(midpoint_timestamp(12.0, 20.0), 16.0)

    def test_invalid_intervals_are_rejected(self):
        for interval in ((2, 2), (3, 1), (float("nan"), 5), ("bad", 3)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                midpoint_timestamp(*interval)

    def test_only_person_class_detections_are_returned(self):
        result = FakeResult(FakeBoxes(
            [[1, 2, 11, 20], [3, 4, 13, 22], [1, 1, 0, 4]],
            [0, 24, 0],
            [0.91, 0.99, 0.7],
        ))
        people = person_detections_from_result(result, confidence_threshold=0.5)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["confidence"], 0.91)

    def test_label_names_yolo_confidence_not_identity(self):
        label = person_detection_label(0.876)
        self.assertEqual(label, "Person | YOLO conf 0.88")
        self.assertNotIn("identity", label.lower())
        self.assertNotIn("match", label.lower())

    def test_box_overlay_returns_rgb_image_with_same_dimensions(self):
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        rendered = draw_person_boxes(frame, [{"xyxy": [-5, 2, 20, 30], "confidence": 0.9}])
        self.assertEqual(rendered.shape, frame.shape)
        self.assertFalse(np.array_equal(rendered, frame))

    def test_missing_detection_boxes_are_empty(self):
        self.assertEqual(person_detections_from_result(FakeResult(None)), [])

    def test_crop_bounds_are_clipped_to_frame(self):
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        frame[0:4, 0:5] = (10, 20, 30)
        crops = crop_person_detections(
            frame,
            [
                {"xyxy": [-4, -3, 5, 4], "confidence": 0.8},
                {"xyxy": [20, 20, 30, 30], "confidence": 0.7},
                {"xyxy": [4, 4, 4, 8], "confidence": 0.6},
            ],
        )
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["xyxy_clipped"], [0, 0, 5, 4])
        self.assertEqual(crops[0]["image_rgb"].shape, (4, 5, 3))
        self.assertEqual(crops[0]["image_rgb"][0, 0].tolist(), [30, 20, 10])

    def test_empty_and_invalid_detections_produce_no_crops(self):
        frame = np.zeros((8, 9, 3), dtype=np.uint8)
        self.assertEqual(crop_person_detections(frame, []), [])
        self.assertEqual(
            crop_person_detections(
                frame,
                [
                    {"xyxy": [float("nan"), 0, 2, 2], "confidence": 0.9},
                    {"xyxy": [20, 20, 30, 30], "confidence": 0.8},
                    {"xyxy": [0, 0, 2, 2], "confidence": float("inf")},
                ],
            ),
            [],
        )

    def test_gallery_items_keep_timestamp_confidence_and_unverified_label(self):
        frame = np.zeros((8, 9, 3), dtype=np.uint8)
        crop = crop_person_detections(
            frame, [{"xyxy": [1, 2, 6, 7], "confidence": 0.876}]
        )
        item = build_person_crop_gallery_items(2, 12.345, crop)[0]
        self.assertEqual(item["candidate_number"], 1)
        self.assertEqual(item["frame_index"], 2)
        self.assertEqual(item["timestamp_seconds"], 12.345)
        self.assertEqual(item["verification_status"], "unverified")
        self.assertIn("12.35s", item["label"])
        self.assertIn("YOLO conf 0.88", item["label"])
        self.assertIn("unverified", item["label"])

    def test_representative_frame_times_are_inside_interval(self):
        times = representative_frame_timestamps(10, 16, max_frames=3)
        self.assertEqual(len(times), 3)
        self.assertTrue(all(10 < value < 16 for value in times))


if __name__ == "__main__":
    unittest.main()
