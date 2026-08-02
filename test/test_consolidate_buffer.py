from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from consolidate_buffer import staging_timestamp  # noqa: E402


class ConsolidationTimestampTests(unittest.TestCase):
    def test_normalizes_legacy_milliseconds_and_current_seconds(self):
        self.assertEqual(
            staging_timestamp("staging/fresh/data_1785041255000_x.jsonl.gz"),
            1785041255,
        )
        self.assertEqual(
            staging_timestamp("staging/worker_1/20260802/batch_1785041255.jsonl.gz"),
            1785041255,
        )

    def test_unknown_names_cannot_advance_watermark(self):
        self.assertEqual(staging_timestamp("staging/unknown.jsonl.gz"), 0)


if __name__ == "__main__":
    unittest.main()
