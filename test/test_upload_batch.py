import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
from upload_batch import merge_and_upload


class UploadSafetyTests(unittest.TestCase):
    def test_zero_completed_games_fail_explicitly(self):
        with patch("upload_batch.glob.glob", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "No completed games"):
                merge_and_upload()
