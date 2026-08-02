from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

import consolidate_buffer  # noqa: E402
from consolidate_buffer import staging_timestamp, validate_buffer, write_size_bounded_buffer  # noqa: E402


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


class ReplayBufferValidationTests(unittest.TestCase):
    def write_buffer(self, records):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "replay.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return temp_dir, path

    def test_accepts_unique_games_in_newest_first_order(self):
        temp_dir, path = self.write_buffer([
            {"id": "new", "ts": 20},
            {"id": "old", "ts": 10},
        ])
        with temp_dir:
            self.assertEqual(validate_buffer(path, 2), 2)

    def test_rejects_duplicate_games(self):
        temp_dir, path = self.write_buffer([
            {"id": "same", "ts": 20},
            {"id": "same", "ts": 10},
        ])
        with temp_dir, self.assertRaisesRegex(RuntimeError, "duplicate game id"):
            validate_buffer(path, 2)

    def test_size_cap_keeps_newest_games_and_drops_oldest(self):
        records = [json.dumps({"id": str(i), "ts": i, "payload": "x" * 200}) for i in range(20, 0, -1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bounded.jsonl.gz"
            original_cap = consolidate_buffer.MAX_REPLAY_BYTES
            try:
                consolidate_buffer.MAX_REPLAY_BYTES = 180
                retained, size = write_size_bounded_buffer(path, records)
            finally:
                consolidate_buffer.MAX_REPLAY_BYTES = original_cap
            self.assertLess(len(retained), len(records))
            self.assertEqual(retained[0], records[0])
            self.assertLessEqual(size, 180)

    def test_rejects_oldest_first_order(self):
        temp_dir, path = self.write_buffer([
            {"id": "old", "ts": 10},
            {"id": "new", "ts": 20},
        ])
        with temp_dir, self.assertRaisesRegex(RuntimeError, "newest-first"):
            validate_buffer(path, 2)


if __name__ == "__main__":
    unittest.main()
