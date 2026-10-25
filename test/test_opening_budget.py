import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from board import DarkChessBoardPy
from search import ChanceSearch, material_evaluate


class OpeningSafetyTests(unittest.TestCase):
    def test_depth_two_requires_time_and_node_budget(self):
        board = DarkChessBoardPy()
        with self.assertRaisesRegex(ValueError, "BOTH"):
            ChanceSearch(evaluator=material_evaluate, max_depth=2).analyze_first_flip(board)

    def test_depth_one_tiny_budget_still_returns_a_legal_flip(self):
        board = DarkChessBoardPy()
        result = ChanceSearch(
            evaluator=material_evaluate, max_depth=1,
            node_budget=1, max_seconds=0.01,
        ).analyze_first_flip(board)
        self.assertEqual(result.depth, 0)
        self.assertIn(result.move, list(map(int, board.generate_legal_moves())))

    def test_depth_one_remains_legal_and_public(self):
        board = DarkChessBoardPy()
        result = ChanceSearch(
            evaluator=material_evaluate, max_depth=1,
            node_budget=1500, max_seconds=5,
        ).analyze_first_flip(board)
        self.assertIn(result.move, list(map(int, board.generate_legal_moves())))
        self.assertEqual(result.depth, 1)
