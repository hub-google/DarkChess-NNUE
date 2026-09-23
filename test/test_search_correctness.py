import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import DarkChessBoardPy, INITIAL_COUNTS, encode_move
from search import ChanceSearch, ExactChanceSearch, material_evaluate


def make_two_hidden_board(last_two):
    inventory = list(np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS))
    for piece in last_two:
        inventory.remove(int(piece))
    bag = np.array(inventory + [int(last_two[0]), int(last_two[1])], dtype=np.int32)
    board = DarkChessBoardPy(bag=bag)
    for square in range(30):
        board.make_move(encode_move(square, square), validate=True)
    return board


class SearchCorrectnessTests(unittest.TestCase):
    def test_star1_root_matches_exact_expectiminimax(self):
        board = make_two_hidden_board((0, 7))
        star = ChanceSearch(
            evaluator=material_evaluate,
            max_depth=2,
        ).analyze(board)
        exact = ExactChanceSearch(
            evaluator=material_evaluate,
            max_depth=2,
        ).analyze(board)

        self.assertEqual(star.move, exact.move)
        self.assertAlmostEqual(star.value, exact.value, places=10)

    def test_hidden_referee_mapping_cannot_change_search(self):
        board_a = make_two_hidden_board((0, 7))
        board_b = make_two_hidden_board((7, 0))

        self.assertEqual(board_a.get_snapshot(), board_b.get_snapshot())
        self.assertNotEqual(
            tuple(board_a.hidden_pieces),
            tuple(board_b.hidden_pieces),
        )

        result_a = ChanceSearch(
            evaluator=material_evaluate,
            max_depth=2,
        ).analyze(board_a)
        result_b = ChanceSearch(
            evaluator=material_evaluate,
            max_depth=2,
        ).analyze(board_b)

        self.assertEqual(result_a.move, result_b.move)
        self.assertEqual(set(result_a.move_values), set(result_b.move_values))
        for move in result_a.move_values:
            self.assertAlmostEqual(
                result_a.move_values[move],
                result_b.move_values[move],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
