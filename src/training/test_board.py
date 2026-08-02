import unittest

import numpy as np

from board import ADVISOR, CHARIOT, RED, DarkChessBoardPy, encode_move, set_bit


class PerpetualChaseTests(unittest.TestCase):
    def setUp(self):
        self.board = DarkChessBoardPy()
        self.board.piece_bitboards[:] = 0
        self.board.hidden_bitboard = np.uint32(0)
        self.board.occupied_bitboard = np.uint32(0)
        self.board.remaining_counts[:] = 0
        self.board.side_to_move = RED

    def move(self, source, target):
        self.board.make_move(int(encode_move(source, target)))

    def test_advancing_chase_is_legal_but_third_route_repetition_is_not(self):
        self.board.piece_bitboards[ADVISOR] = set_bit(self.board.piece_bitboards[ADVISOR], 8)
        self.board.piece_bitboards[7 + CHARIOT] = set_bit(self.board.piece_bitboards[7 + CHARIOT], 1)
        self.board.occupied_bitboard = np.uint32((1 << 8) | (1 << 1))
        route = (
            (8, 0),
            (1, 9), (0, 1), (9, 8), (1, 9), (8, 0), (9, 8), (0, 1),
        )
        for source, target in route:
            self.move(source, target)

        forbidden = int(encode_move(8, 0))
        legal = set(map(int, self.board.generate_legal_moves()))
        self.assertNotIn(forbidden, legal)
        self.assertNotIn(int(encode_move(8, 9)), legal)
        self.assertIn(int(encode_move(8, 16)), legal)
        with self.assertRaises(ValueError):
            self.board.make_move(forbidden, validate=True)

    def test_threefold_repetition_is_not_a_draw(self):
        self.board.piece_bitboards[CHARIOT] = set_bit(self.board.piece_bitboards[CHARIOT], 0)
        self.board.piece_bitboards[7 + CHARIOT] = set_bit(self.board.piece_bitboards[7 + CHARIOT], 31)
        self.board.occupied_bitboard = np.uint32(1 | (1 << 31))
        for source, target in (
            (0, 1), (31, 30), (1, 0), (30, 31),
            (0, 1), (31, 30), (1, 0), (30, 31),
        ):
            self.move(source, target)
        self.assertEqual(self.board.is_game_over(), (False, 0.0))


if __name__ == '__main__':
    unittest.main()
