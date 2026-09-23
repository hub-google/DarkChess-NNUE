import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import DarkChessBoardPy, INITIAL_COUNTS, decode_move
from nnue_eval import ModelEvaluator, feature_deltas
from train import DarkChessNNUE, extract_features


class IncrementalNNUETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DarkChessNNUE()
        self.model.eval()
        self.evaluator = ModelEvaluator(self.model)
        bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        np.random.default_rng(1234).shuffle(bag)
        self.board = DarkChessBoardPy(bag=bag)

    def full_value(self, board):
        features = torch.from_numpy(extract_features(board)).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(features).item())

    def test_sparse_feature_delta_reconstructs_child_features(self):
        board = self.board
        rng = np.random.default_rng(11)
        for _ in range(30):
            moves = [int(move) for move in board.generate_legal_moves()]
            if not moves:
                break
            move = int(rng.choice(moves))
            from_sq, _, is_flip = decode_move(move)
            flip_piece = int(board.hidden_pieces[from_sq]) if is_flip else None
            child = board.clone()
            child.make_move(move, flip_piece=flip_piece, validate=False)

            reconstructed = extract_features(board).copy()
            for index, delta in feature_deltas(
                board,
                child,
                move,
                flip_piece=flip_piece,
            ):
                reconstructed[index] += delta
            np.testing.assert_allclose(
                reconstructed,
                extract_features(child),
                rtol=0.0,
                atol=1e-7,
            )
            board = child

    def test_in_place_move_invalidates_stale_root_accumulator(self):
        board = self.board
        self.evaluator(board)
        move = int(board.generate_legal_moves()[0])
        from_sq, _, is_flip = decode_move(move)
        flip_piece = int(board.hidden_pieces[from_sq]) if is_flip else None
        board.make_move(move, flip_piece=flip_piece, validate=False)

        self.assertAlmostEqual(
            self.evaluator(board),
            self.full_value(board),
            places=5,
        )


    def test_fully_revealed_position_uses_full_forward_fallback(self):
        board = self.board
        while int(board.hidden_bitboard) != 0:
            flips = [
                int(move)
                for move in board.generate_legal_moves()
                if decode_move(int(move))[2]
            ]
            if not flips:
                self.fail("expected a legal flip while hidden pieces remain")
            move = flips[0]
            from_sq, _, _ = decode_move(move)
            board.make_move(
                move,
                flip_piece=int(board.hidden_pieces[from_sq]),
                validate=False,
            )

        self.assertFalse(self.evaluator._use_incremental(board))
        self.assertAlmostEqual(
            self.evaluator(board),
            self.full_value(board),
            places=6,
        )

    def test_incremental_matches_full_forward_through_random_game(self):
        board = self.board
        rng = np.random.default_rng(99)
        checked = 0
        for _ in range(60):
            self.assertAlmostEqual(
                self.evaluator(board),
                self.full_value(board),
                places=5,
            )
            checked += 1
            over, _ = board.is_game_over()
            if over:
                break
            moves = [int(move) for move in board.generate_legal_moves()]
            if not moves:
                break
            move = int(rng.choice(moves))
            from_sq, _, is_flip = decode_move(move)
            flip_piece = int(board.hidden_pieces[from_sq]) if is_flip else None
            child = board.clone()
            child.make_move(move, flip_piece=flip_piece, validate=False)
            self.evaluator.prepare_child(
                board,
                child,
                move,
                flip_piece=flip_piece,
            )
            board = child

        self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
    unittest.main()
