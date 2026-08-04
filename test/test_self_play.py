import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS_DIR = PROJECT_ROOT / "src" / "workers"
sys.path.insert(0, str(WORKERS_DIR))

try:
    import self_play
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    self_play = None


@unittest.skipIf(self_play is None, "torch is not installed in this test environment")
class AdaptiveDepthTests(unittest.TestCase):
    def test_uses_fixed_hidden_piece_depth_bands(self):
        expected = {
            32: 3,
            24: 3,
            23: 10,
            12: 10,
            11: 12,
            0: 12,
        }
        for hidden_count, depth in expected.items():
            with self.subTest(hidden_count=hidden_count):
                self.assertEqual(self_play.choose_search_depth(hidden_count), depth)


@unittest.skipIf(self_play is None, "torch is not installed in this test environment")
class IncrementalPersistenceTests(unittest.TestCase):
    def test_saves_each_game_before_starting_the_next_one(self):
        games = [
            {"id": "game-one", "mov": [1], "q": [0.0], "res": 1.0, "ply": 1},
            {"id": "game-two", "mov": [2], "q": [0.0], "res": -1.0, "ply": 1},
        ]
        observed_file_counts = []

        def fake_play_game(*args, **kwargs):
            observed_file_counts.append(len(list(output_dir.glob("*.jsonl.gz"))))
            return games[len(observed_file_counts) - 1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(self_play, "play_game", side_effect=fake_play_game):
                self_play.run_batch(2, output_dir, None, "test-model", None)

            self.assertEqual(observed_file_counts, [0, 1])
            files = sorted(output_dir.glob("*.jsonl.gz"))
            self.assertEqual(len(files), 2)
            saved_ids = []
            for path in files:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    saved_ids.append(json.loads(handle.readline())["id"])
            self.assertCountEqual(saved_ids, ["game-one", "game-two"])


if __name__ == "__main__":
    unittest.main()
