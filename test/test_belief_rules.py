import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import (  # noqa: E402
    BLACK,
    INITIAL_COUNTS,
    RED,
    DarkChessBoardPy,
    encode_move,
)
from search import ChanceSearch, material_evaluate  # noqa: E402


class BeliefAndRuleTests(unittest.TestCase):
    def test_initial_belief_matches_inventory(self):
        board = DarkChessBoardPy()
        np.testing.assert_array_equal(board.remaining_counts, INITIAL_COUNTS)
        self.assertAlmostEqual(board.hidden_probability(0), 1 / 32)
        self.assertAlmostEqual(board.hidden_probability(6), 5 / 32)

    def test_flip_updates_public_count_and_assigns_color(self):
        bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        board = DarkChessBoardPy(bag=bag)
        board.make_move(encode_move(0, 0))
        self.assertEqual(board.remaining_counts[0], 0)
        self.assertEqual(board.side_to_move, BLACK)

    def test_color_with_no_remaining_or_visible_piece_loses_immediately(self):
        board = DarkChessBoardPy()
        board.piece_bitboards.fill(0)
        board.occupied_bitboard = np.uint32(0)
        board.hidden_bitboard = np.uint32(1)
        board.remaining_counts.fill(0)
        board.remaining_counts[13] = 1
        board.hidden_pieces.fill(13)
        board.side_to_move = RED
        self.assertEqual(board.is_game_over(), (True, -1.0))

    def test_sixty_quiet_plies_is_a_draw(self):
        board = DarkChessBoardPy()
        board.half_move_clock = 60
        self.assertEqual(board.is_game_over(), (True, 0.0))

    def test_illegal_move_is_rejected(self):
        board = DarkChessBoardPy()
        with self.assertRaisesRegex(ValueError, "illegal move"):
            board.make_move(encode_move(0, 1))

    def test_search_does_not_depend_on_true_hidden_mapping(self):
        bag_a = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        bag_b = bag_a[::-1].copy()
        board_a = DarkChessBoardPy(bag=bag_a)
        board_b = DarkChessBoardPy(bag=bag_b)

        opening_a = ChanceSearch(material_evaluate, max_depth=1).analyze_first_flip(board_a)
        opening_b = ChanceSearch(material_evaluate, max_depth=1).analyze_first_flip(board_b)
        self.assertEqual(opening_a.move_values, opening_b.move_values)

        # Give both public states the same observed first flip, even though
        # their private referee mappings differ.
        board_a.make_move(encode_move(0, 0), flip_piece=0, validate=False)
        board_b.make_move(encode_move(0, 0), flip_piece=0, validate=False)
        board_a.hidden_pieces.fill(-999)
        board_b.hidden_pieces.fill(999)

        result_a = ChanceSearch(material_evaluate, max_depth=1).analyze(board_a)
        result_b = ChanceSearch(material_evaluate, max_depth=1).analyze(board_b)
        self.assertEqual(result_a.move, result_b.move)
        self.assertEqual(result_a.move_values, result_b.move_values)


if __name__ == "__main__":
    unittest.main()
