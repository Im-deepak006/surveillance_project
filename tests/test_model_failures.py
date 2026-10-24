import os
import tempfile
import unittest

from run_on_video.model_utils import build_inference_model

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "run_on_video",
    "moment_detr_ckpt",
    "model_best.ckpt",
)


class ModelFailureTests(unittest.TestCase):
    def test_missing_checkpoint_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "missing_checkpoint.ckpt")
            with self.assertRaises(FileNotFoundError):
                build_inference_model(missing)

    @unittest.skipUnless(os.path.isfile(CHECKPOINT), "local Moment-DETR checkpoint is unavailable")
    def test_inference_model_disables_dropout(self):
        model = build_inference_model(CHECKPOINT)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
