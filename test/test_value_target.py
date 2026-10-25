import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from train import value_target, select_training_plies


class SafeTeacherValueTests(unittest.TestCase):
    def test_depth_zero_uses_result_only(self):
        self.assertEqual(value_target(1.0, -1.0, search_depth=0), 1.0)
        self.assertEqual(value_target(1.0, -1.0, search_depth=2), 0.0)
        self.assertEqual(value_target(1.0, -1.0, search_depth=2, search_weight=0.25), 0.5)

    def test_bad_weight_rejected(self):
        with self.assertRaises(ValueError):
            value_target(0.0, 0.5, 2, -0.1)

    def test_fallback_is_not_eligible_as_hard_position(self):
        game = {"id": "x", "q": [0, 0.9, -0.7, 0.2, 0.1, 0],
                "v": [0, -0.9, 0.7, 0, 0, 0], "qd": [1, 0, 0, 2, 2, 1]}
        plies = list(select_training_plies(game, list(range(6)), 2))
        self.assertNotIn(1, plies)
        self.assertNotIn(2, plies)
