import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from validate_training_readiness import check_readiness


class TrainingReadinessTests(unittest.TestCase):
    def test_tiny_smoke_cannot_promote(self):
        with self.assertRaisesRegex(RuntimeError, "refusing to train"):
            check_readiness({"valid_games": 2, "samples": 8}, min_games=1000)

    def test_minimum_valid_replay_works(self):
        self.assertTrue(check_readiness(
            {"valid_games": 1000, "samples": 4000}, min_games=1000,
        ))

    def test_too_few_samples_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Too few samples"):
            check_readiness({"valid_games": 1000, "samples": 999}, min_games=1000)
