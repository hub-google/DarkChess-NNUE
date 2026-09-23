import sys
import unittest
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import DarkChessBoardPy, RED, encode_move
from tablebase import (
    EndgameTablebase,
    canonical_key,
    estimate_raw_states,
    make_test_board,
)


class EndgameTablebaseTests(unittest.TestCase):
    def test_two_piece_forced_capture_is_exact(self):
        board = make_test_board([(6, 0), (7, 1)], RED)
        tablebase = EndgameTablebase(max_pieces=4, max_states=10000)
        result = tablebase.analyze(board)

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.distance, 1)
        self.assertEqual(result.best_move, encode_move(0, 1))

    def test_horizontal_symmetry_has_same_canonical_key(self):
        left = make_test_board([(6, 0), (7, 1)], RED)
        right = make_test_board([(6, 7), (7, 6)], RED)
        self.assertEqual(canonical_key(left), canonical_key(right))

    def test_rejects_hidden_positions(self):
        with self.assertRaises(ValueError):
            EndgameTablebase().probe(DarkChessBoardPy())

    def test_state_estimate_grows_quickly(self):
        two = estimate_raw_states(2)
        three = estimate_raw_states(3)
        four = estimate_raw_states(4)
        self.assertGreater(two, 0)
        self.assertGreater(three, two)
        self.assertGreater(four, three)


if __name__ == "__main__":
    unittest.main()
