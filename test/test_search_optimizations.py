import sys
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import DarkChessBoardPy, INITIAL_COUNTS
from search import ChanceSearch, material_evaluate


class SearchOptimizationTests(unittest.TestCase):
    def make_board(self):
        bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        board = DarkChessBoardPy(bag=bag)
        board.make_move(0, validate=True)
        return board

    def test_node_budget_returns_completed_iteration(self):
        board = self.make_board()
        result = ChanceSearch(
            evaluator=material_evaluate,
            max_depth=8,
            node_budget=500,
        ).analyze(board)
        self.assertIn(result.move, [int(m) for m in board.generate_legal_moves()])
        self.assertGreaterEqual(result.depth, 1)
        self.assertLessEqual(result.depth, 8)

    def test_unbudgeted_search_reports_requested_depth(self):
        board = self.make_board()
        result = ChanceSearch(
            evaluator=material_evaluate,
            max_depth=2,
        ).analyze(board)
        self.assertEqual(result.depth, 2)


if __name__ == "__main__":
    unittest.main()
