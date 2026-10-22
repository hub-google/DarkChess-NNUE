import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from board import DarkChessBoardPy
from misty_uci import public_fen, move_to_uci, uci_to_move, choose_misty_move


class MistyAdapterTests(unittest.TestCase):
    def test_opening_fen_is_reference_compatible(self):
        board = DarkChessBoardPy()
        self.assertEqual(
            public_fen(board),
            "XXXXXXXX/XXXXXXXX/XXXXXXXX/XXXXXXXX - G1A2E2R2H2C2S5g1a2e2r2h2c2s5 0 1",
        )

    def test_never_export_hidden_referee_mapping(self):
        board = DarkChessBoardPy()
        board.make_move(0, validate=True)
        clone = board.clone()
        clone.hidden_pieces[:] = clone.hidden_pieces[::-1]
        self.assertEqual(public_fen(board), public_fen(clone))

    def test_uci_move_round_trip(self):
        for square in (0, 7, 9, 31):
            move = (square << 5) | square
            self.assertEqual(uci_to_move(move_to_uci(move)), move)
        self.assertEqual(uci_to_move("a0h3"), 31)

    def test_engine_protocol_is_bounded_and_legal(self):
        board = DarkChessBoardPy()
        board.make_move(0, validate=True)
        allowed = [int(m) for m in board.generate_legal_moves()]
        token = move_to_uci(allowed[0])
        with patch("misty_uci.subprocess.run", return_value=SimpleNamespace(stdout=f"uciok\nreadyok\nbestmove {token}\n")) as mock:
            move = choose_misty_move(board, "/fake/misty", node_budget=250, max_seconds=0.1)
            self.assertEqual(move, allowed[0])
            command = mock.call_args.kwargs["input"]
            self.assertIn("setoption name NoProgressLimit value 60", command)
            self.assertIn("searchmoves", command)
            self.assertIn("go nodes 250 movetime 100", command)

    def test_illegal_engine_move_is_rejected(self):
        board = DarkChessBoardPy()
        board.make_move(0, validate=True)
        with patch("misty_uci.subprocess.run", return_value=SimpleNamespace(stdout="bestmove a0a0\n")):
            with self.assertRaisesRegex(RuntimeError, "illegal"):
                choose_misty_move(board, "/fake/misty")
