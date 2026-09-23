import sys
import unittest
from pathlib import Path

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from train import (
    CURRENT_INPUT_SIZE,
    SYM_LEFT_RIGHT,
    SYM_ROTATE_180,
    SYM_UP_DOWN,
    augment_features,
    transform_square,
)


class TrainingAugmentationTests(unittest.TestCase):
    def test_rectangle_has_eight_geometric_opening_classes(self):
        representatives = {
            min(
                square,
                transform_square(square, SYM_LEFT_RIGHT),
                transform_square(square, SYM_UP_DOWN),
                transform_square(square, SYM_ROTATE_180),
            )
            for square in range(32)
        }
        self.assertEqual(len(representatives), 8)

    def test_rotate_180_moves_square_features_without_changing_value(self):
        features = np.zeros(CURRENT_INPUT_SIZE, dtype=np.float32)
        features[0 * 15 + 3] = 1.0
        features[10 * 15 + 14] = 1.0
        features[494] = 1.0

        transformed, target = augment_features(
            features,
            0.625,
            transform=SYM_ROTATE_180,
            color_swap=False,
        )

        self.assertEqual(transformed[31 * 15 + 3], 1.0)
        self.assertEqual(transformed[21 * 15 + 14], 1.0)
        self.assertEqual(transformed[494], 1.0)
        self.assertAlmostEqual(target, 0.625)

    def test_color_swap_swaps_piece_counts_and_side_and_negates_target(self):
        features = np.zeros(CURRENT_INPUT_SIZE, dtype=np.float32)
        features[5 * 15 + 0] = 1.0
        features[480] = 0.75
        features[487] = 0.25
        features[494] = 1.0
        features[496] = 0.4

        transformed, target = augment_features(
            features,
            0.4,
            color_swap=True,
        )

        self.assertEqual(transformed[5 * 15 + 7], 1.0)
        self.assertEqual(transformed[480], 0.25)
        self.assertEqual(transformed[487], 0.75)
        self.assertEqual(transformed[494], 0.0)
        self.assertEqual(transformed[495], 1.0)
        self.assertAlmostEqual(float(transformed[496]), 0.4, places=6)
        self.assertAlmostEqual(target, -0.4)


if __name__ == "__main__":
    unittest.main()
