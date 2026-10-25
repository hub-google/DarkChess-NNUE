import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))

from train import CURRENT_INPUT_SIZE, DarkChessNNUE, augment_features, extract_features, transform_square
from board import DarkChessBoardPy, INITIAL_COUNTS
from policy_train import transform_actions, transform_legal_mask
from symmetry_eval import symmetry_metrics


def transformed_bag(bag, transform, color_swap):
    out = np.empty(32, dtype=np.int32)
    for square, piece in enumerate(bag):
        identity = (int(piece) + 7) % 14 if color_swap else int(piece)
        out[transform_square(square, transform)] = identity
    return out


class SymmetryRulesTests(unittest.TestCase):
    def test_live_game_replay_preserves_legal_moves_and_public_state(self):
        # This tests the referee, history/repetition and forbidden-chase state,
        # not merely one-hot feature permutations.
        bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        rng = np.random.default_rng(20261024)
        rng.shuffle(bag)
        board = DarkChessBoardPy(bag=bag)
        variants = [(sym, swap, DarkChessBoardPy(bag=transformed_bag(bag, sym, swap)))
                    for swap in (False, True) for sym in range(4)]
        for _ in range(65):
            moves = sorted(map(int, board.generate_legal_moves()))
            result = board.is_game_over()
            features = extract_features(board)
            for sym, swap, equivalent in variants:
                actual = sorted(map(int, equivalent.generate_legal_moves()))
                expected = sorted(map(int, transform_actions(np.array(moves), sym)))
                self.assertEqual(actual, expected, (sym, swap))
                transformed, _ = augment_features(features, 0.0, sym, swap)
                np.testing.assert_array_equal(transformed, extract_features(equivalent))
                self.assertEqual(equivalent.repetition_count(), board.repetition_count())
                alternate_result = equivalent.is_game_over()
                self.assertEqual(result[0], alternate_result[0])
                self.assertEqual(alternate_result[1], -result[1] if swap else result[1])
                if moves:
                    mask = np.zeros(128, dtype=np.uint8)
                    for move in moves:
                        mask[move // 8] |= 1 << (move % 8)
                    self.assertEqual(set(np.flatnonzero(transform_legal_mask(mask, sym))),
                                     set(actual))
            if result[0] or not moves:
                break
            move = moves[int(rng.integers(len(moves)))]
            board.make_move(move, validate=True)
            for sym, _, equivalent in variants:
                equivalent.make_move(int(transform_actions([move], sym)[0]), validate=True)


class SymmetryModelTests(unittest.TestCase):
    def test_exact_zero_model_has_no_symmetry_error(self):
        model = DarkChessNNUE()
        for param in model.parameters():
            torch.nn.init.zeros_(param)
        bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        board = DarkChessBoardPy(bag=bag)
        features = np.stack([extract_features(board)] * 4)
        metrics = symmetry_metrics(model, features)
        self.assertEqual(metrics["positions"], 4)
        self.assertEqual(len(metrics["transforms"]), 7)
        self.assertEqual(metrics["overall_mean_abs_error"], 0.0)

    def test_asymmetric_model_has_measurable_error(self):
        class AsymmetricValue(torch.nn.Module):
            input_size = CURRENT_INPUT_SIZE

            def forward(self, x):
                return 0.5 * x[:, 0:1] + 0.1

        features = np.zeros((4, CURRENT_INPUT_SIZE), dtype=np.float32)
        features[:, 0] = 1
        metrics = symmetry_metrics(AsymmetricValue(), features)
        self.assertGreater(metrics["geometry_mean_abs_error"], 0.1)
        self.assertGreater(metrics["color_mean_abs_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
