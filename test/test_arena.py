import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from arena import elo_from_score, paired_benchmark


class ArenaTests(unittest.TestCase):
    def test_elo_direction_and_neutral(self):
        self.assertAlmostEqual(elo_from_score(0.5), 0.0)
        self.assertGreater(elo_from_score(0.6), 0.0)
        self.assertLess(elo_from_score(0.4), 0.0)

    def test_balanced_pair_with_identical_players(self):
        with patch("arena.play_game", side_effect=[0.5, 0.5, 0.5, 0.5]):
            result = paired_benchmark(lambda board: 0., lambda board: 0.,
                                      pairs=2, depth=1, node_budget=2)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["paired_games"], 4)
        self.assertEqual(result["candidate_points"], 2.0)
        self.assertAlmostEqual(result["candidate_score_rate"], 0.5)

    def test_invalid_limits_fail(self):
        with self.assertRaises(ValueError):
            paired_benchmark(None, None, pairs=0)
