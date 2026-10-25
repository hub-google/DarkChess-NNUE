import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from build_training_cache import write_reanalysis_overrides
from train import CURRENT_INPUT_SIZE, TrainingCacheDataset


class TrainingCacheTests(unittest.TestCase):
    def test_binary_cache_applies_reanalysis_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            features = np.zeros((2, CURRENT_INPUT_SIZE), dtype=np.float16)
            features[0, 0] = 1.0
            features[1, 15] = 1.0
            np.savez_compressed(
                cache_dir / "cache_0000.npz",
                features=features,
                targets=np.asarray([0.25, -0.5], dtype=np.float32),
                sample_index=np.asarray([10, 11], dtype=np.int64),
            )
            write_reanalysis_overrides(
                cache_dir / "reanalysis_overrides.npz",
                [(11, 0.75, 0.5, 0.1, 0.4, 100, 3)],
            )

            dataset = TrainingCacheDataset(
                str(cache_dir),
                max_samples=2,
                symmetry_augmentation=False,
                color_swap_augmentation=False,
            )
            rows = list(dataset)
            observed = {}
            for feature, target in rows:
                active = int(np.flatnonzero(feature.numpy()[:30])[0])
                observed[active] = float(target.item())

            self.assertAlmostEqual(observed[0], 0.25)
            self.assertAlmostEqual(observed[15], 0.75)

    def test_empty_reanalysis_override_file_is_loadable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "overrides.npz"
            write_reanalysis_overrides(path, [])
            with np.load(path) as data:
                self.assertEqual(len(data["sample_index"]), 0)
                self.assertEqual(len(data["target"]), 0)


if __name__ == "__main__":
    unittest.main()
