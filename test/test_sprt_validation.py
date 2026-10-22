import sys
import unittest
import json
import tempfile
import numpy as np
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from sprt_validation import elo_probability, paired_score_llr, play_game, play_completed_pair
from unittest.mock import patch
from types import SimpleNamespace


class PairedSPRTTests(unittest.TestCase):
    def test_incomplete_paired_deal_does_not_fake_draw(self):
        incomplete = RuntimeError("SPRT referee could not reach a legal terminal within 512 plies")
        # The first matched deal fails in its second game. It must be dropped
        # completely; only the second deal is counted as a full valid pair.
        with patch("sprt_validation.play_game", side_effect=[1.0, incomplete, 0.5, 0.0]):
            points, discarded = play_completed_pair(
                None, None, np.random.default_rng(3), 1, 50, 0.1
            )
        self.assertEqual(points, 1.5)
        self.assertEqual(discarded, 1)

    def test_all_unfinished_games_refuse_promotion(self):
        incomplete = RuntimeError("SPRT referee could not reach a legal terminal within 512 plies")
        with patch("sprt_validation.play_game", side_effect=incomplete):
            with self.assertRaisesRegex(RuntimeError, "Unable to complete"):
                play_completed_pair(None, None, np.random.default_rng(3), 1, 50, 0.1, max_attempts=2)


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
