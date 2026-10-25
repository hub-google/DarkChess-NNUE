import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from search import PIECE_VALUES, material_evaluate
from external_arena import HybridEvaluator
from board import DarkChessBoardPy


class MaterialOrderingTests(unittest.TestCase):
    def test_cannon_value_exceeds_chariot_value(self):
        self.assertGreater(PIECE_VALUES[5], PIECE_VALUES[3])
        self.assertEqual(PIECE_VALUES[5], PIECE_VALUES[12])

    def test_hybrid_is_bounded_and_public(self):
        board = DarkChessBoardPy()
        result = HybridEvaluator(lambda _: 0.4, material_weight=0.25)(board)
        self.assertLessEqual(abs(result), 1.0)
        mirror = board.clone()
        mirror.hidden_pieces[:] = mirror.hidden_pieces[::-1]
        self.assertAlmostEqual(
            result,
            HybridEvaluator(lambda _: 0.4, material_weight=0.25)(mirror),
        )
