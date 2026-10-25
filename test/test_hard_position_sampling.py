import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from train import select_training_plies


class HardPositionSamplingTests(unittest.TestCase):
    def test_selects_large_search_disagreements_and_random_coverage(self):
        moves = list(range(10))
        game = {
            "id": "stable-test-id",
            "q": [0.0, 0.9, 0.1, -0.8, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4],
            "v": [0.0, -0.9, 0.1, 0.8, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4],
        }
        selected = set(int(x) for x in select_training_plies(game, moves, 4))
        self.assertEqual(len(selected), 4)
        self.assertIn(1, selected)
        self.assertIn(3, selected)

    def test_falls_back_for_legacy_replay_without_v(self):
        moves = list(range(10))
        game = {"id": "legacy", "q": [0.0] * 10}
        selected = select_training_plies(game, moves, 4)
        self.assertEqual(len(selected), 4)


if __name__ == "__main__":
    unittest.main()
