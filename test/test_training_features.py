import sys
from pathlib import Path
import gzip
import json
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import BLACK, INITIAL_COUNTS, RED, DarkChessBoardPy  # noqa: E402


try:
    import torch
    from train import (  # noqa: E402
        CURRENT_INPUT_SIZE,
        LEGACY_INPUT_SIZE,
        DarkChessNNUE,
        DarkChessDataset,
        extract_features,
        initialize_challenger,
    )
except ImportError:
    CURRENT_INPUT_SIZE = None
    extract_features = None


@unittest.skipIf(extract_features is None, "PyTorch is not installed")
class TrainingFeatureTests(unittest.TestCase):
    def test_features_use_public_counts_not_private_mapping(self):
        bag_a = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
        bag_b = bag_a[::-1].copy()
        board_a = DarkChessBoardPy(bag=bag_a)
        board_b = DarkChessBoardPy(bag=bag_b)
        np.testing.assert_array_equal(
            extract_features(board_a),
            extract_features(board_b),
        )

    def test_side_and_draw_state_are_encoded(self):
        board = DarkChessBoardPy()
        board.side_to_move = RED
        red_features = extract_features(board)
        board.side_to_move = BLACK
        board.half_move_clock = 30
        black_features = extract_features(board)

        self.assertEqual(len(red_features), CURRENT_INPUT_SIZE)
        self.assertEqual(red_features[494], 1.0)
        self.assertEqual(red_features[495], 0.0)
        self.assertEqual(black_features[494], 0.0)
        self.assertEqual(black_features[495], 1.0)
        self.assertAlmostEqual(black_features[496], 0.5)

    def test_legacy_champion_is_upgraded_without_discarding_weights(self):
        legacy = DarkChessNNUE(input_size=LEGACY_INPUT_SIZE)
        with torch.no_grad():
            legacy.fc1.weight.fill_(0.25)
            legacy.fc1.bias.fill_(0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.nnue"
            torch.save(legacy.state_dict(), path)
            upgraded = initialize_challenger(path)

        self.assertEqual(upgraded.input_size, CURRENT_INPUT_SIZE)
        self.assertTrue(
            torch.allclose(
                upgraded.fc1.weight[:, 0],
                legacy.fc1.weight[:, 0],
            )
        )
        self.assertTrue(torch.count_nonzero(upgraded.fc1.weight[:, 494:]) == 0)

    def test_v1_always_capture_games_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"ver": "v1.0.0-heuristic"}) + "\n")
            self.assertEqual(list(DarkChessDataset([str(path)])), [])


if __name__ == "__main__":
    unittest.main()
