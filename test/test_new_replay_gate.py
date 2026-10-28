"""Regression tests for the no-new-replay training gate."""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from new_replay_gate import has_new_games, snapshot_replays


class ReplayTrainingGateTests(unittest.TestCase):
    def make_snapshot(self, games):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        path = Path(self.temp_dir.name) / "replay.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as writer:
            for game_id, timestamp in games:
                writer.write(json.dumps({"id": game_id, "ts": timestamp}) + "\n")
        return snapshot_replays(path)

    def test_unchanged_replay_must_skip_training(self):
        current = self.make_snapshot([("a", 1000), ("b", 2000)])
        self.assertFalse(has_new_games(current, current))

    def test_new_id_increases_game_count_and_trains(self):
        previous = self.make_snapshot([("a", 1000)])
        current = self.make_snapshot([("a", 1000), ("b", 2000)])
        self.assertTrue(has_new_games(current, previous))

    def test_replaced_id_at_capacity_trains(self):
        previous = self.make_snapshot([("a", 1000), ("b", 2000)])
        current = self.make_snapshot([("b", 2000), ("c", 3000)])
        self.assertTrue(has_new_games(current, previous))

    def test_bootstrap_recorded_count_skips_old_games(self):
        current = self.make_snapshot([("a", 1000), ("b", 2000)])
        initial = {"total_games": 2, "ids_sha256": None, "newest_ts": None}
        self.assertFalse(has_new_games(current, initial))

    def test_bootstrap_detects_increased_count(self):
        current = self.make_snapshot([("a", 1000), ("b", 2000)])
        initial = {"total_games": 1, "ids_sha256": None, "newest_ts": None}
        self.assertTrue(has_new_games(current, initial))

    def test_missing_marker_fails_closed(self):
        current = self.make_snapshot([("a", 1000)])
        with self.assertRaises(RuntimeError):
            has_new_games(current, None)

    def test_duplicate_game_id_is_invalid(self):
        with self.assertRaises(ValueError):
            self.make_snapshot([("a", 1000), ("a", 1000)])

    def test_empty_buffer_is_invalid(self):
        with self.assertRaises(ValueError):
            self.make_snapshot([])


if __name__ == "__main__":
    unittest.main()
