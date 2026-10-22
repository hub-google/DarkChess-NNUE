import sys
import unittest
import json
import tempfile
from pathlib import Path
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from sprt_validation import elo_probability, paired_score_llr, play_game
from unittest.mock import patch
from types import SimpleNamespace


class PairedSPRTTests(unittest.TestCase):
    def test_nonterminal_after_move_limit_must_not_fake_draw(self):
        class NeverTerminal:
            hidden_pieces = [0] * 32
            side_to_move = 0
            def make_move(self, *args, **kwargs):
                return None
            def is_game_over(self):
                return False, 0.0

        with patch("sprt_validation.DarkChessBoardPy", return_value=NeverTerminal()), patch(
            "sprt_validation.ChanceSearch"
        ) as search:
            search.return_value.analyze.return_value = SimpleNamespace(move=0)
            with self.assertRaisesRegex(RuntimeError, "refusing to silently count"):
                play_game(lambda _: 0., lambda _: 0., [0] * 32, 0, 1, 1)
    def test_elo_probability_is_monotone(self):
        self.assertLess(elo_probability(-15), elo_probability(0))
        self.assertLess(elo_probability(0), elo_probability(15))

    def test_symmetric_elo_bounds_make_split_pair_neutral(self):
        llr = paired_score_llr(1.0, -15.0, 15.0)
        self.assertAlmostEqual(llr, 0.0, places=12)

    def test_pair_score_moves_llr_in_expected_direction(self):
        loss = paired_score_llr(0.0, -15.0, 15.0)
        split = paired_score_llr(1.0, -15.0, 15.0)
        win = paired_score_llr(2.0, -15.0, 15.0)
        self.assertLess(loss, split)
        self.assertLess(split, win)

    def test_llr_adds_sequentially(self):
        one = paired_score_llr(2.0, -15.0, 15.0)
        self.assertAlmostEqual(
            one + one,
            2.0 * paired_score_llr(2.0, -15.0, 15.0),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
