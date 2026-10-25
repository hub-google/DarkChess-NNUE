import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from replay_gate import count_completed_replays, write_github_output


class ReplayGateTests(unittest.TestCase):
    def test_not_ready_and_ready(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "replays.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write('{"id":"1"}\n{"id":"2"}\n')
            self.assertEqual(count_completed_replays(path, 3), (2, False))
            self.assertEqual(count_completed_replays(path, 2), (2, True))
            output_file = Path(dirname) / "result.txt"
            write_github_output(output_file, 2, False)
            self.assertEqual(output_file.read_text(), "ready=false\nreplay_games=2\n")

    def test_bad_minimum(self):
        with self.assertRaises(ValueError):
            count_completed_replays("missing.gz", 0)
