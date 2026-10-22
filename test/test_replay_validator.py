import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from validate_replay import validate_record


class ReplayValidationTests(unittest.TestCase):
    def test_missing_qd_length_rejected(self):
        invalid = {"ver": "v2.1.0-perpetual-chase", "mov": [1, 2], "qd": [1]}
        with self.assertRaisesRegex(ValueError, "qd length"):
            validate_record(invalid)

    def test_fallback_must_not_have_policy_teacher(self):
        from board import DarkChessBoardPy
        board = DarkChessBoardPy()
        from board import encode_move
        first = int(encode_move(0, 0))
        board.make_move(first, validate=True)
        # Incomplete replays must never be published as valid.
        with self.assertRaises(ValueError):
            validate_record({"ver": "v2.1.0-perpetual-chase", "mov": [first], "hid": board.hidden_pieces.tolist(),
                             "res": 0, "qd": [0], "pi": [[[first, 1.0]]]})
