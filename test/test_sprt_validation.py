import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from sprt_validation import elo_probability, paired_score_llr


class PairedSPRTTests(unittest.TestCase):
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
