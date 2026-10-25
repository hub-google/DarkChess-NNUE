import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
from upload_batch import merge_and_upload, MAX_ATTEMPTS

class UploadSafetyTests(unittest.TestCase):
    def test_missing_artifacts_fail(self):
        with patch("upload_batch.glob.glob", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "No completed games"):
                merge_and_upload()

    def test_hf_failure_never_deletes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data_1785000000_example.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as out:
                out.write(json.dumps({"id":"example","ts":1785000000000}) + "\n")
            mock_api = MagicMock()
            mock_api.file_exists.return_value = False
            mock_api.upload_file.side_effect = RuntimeError("429 rate limited")
            with (
                patch.dict(os.environ, {"OUTPUT_DIR":directory,"HF_TOKEN":"fake"}),
                patch("upload_batch.validate_record"),
                patch("upload_batch.HfApi", return_value=mock_api),
                patch("upload_batch.time.sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "HF upload unverified"):
                    merge_and_upload()
            self.assertTrue(source.exists())
            self.assertEqual(mock_api.upload_file.call_count, MAX_ATTEMPTS)

    def test_fifteen_workers_have_exactly_one_hf_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for worker in range(1, 16):
                source = root / f"worker_{worker}.jsonl.gz"
                with gzip.open(source, "wt", encoding="utf-8") as out:
                    out.write(json.dumps({
                        "id": f"worker-game-{worker}",
                        "ts": 1785000000000 + worker,
                    }) + "\n")
            api = MagicMock()
            api.file_exists.return_value = False

            def downloaded_remote(**kwargs):
                # Stand in for the verified HF bytes after the single upload.
                archive = list(root.glob("batch_*.jsonl.gz"))
                self.assertEqual(1, len(archive))
                return str(archive[0])

            with (
                patch.dict(os.environ, {"OUTPUT_DIR": directory, "HF_TOKEN": "fake", "WAVE": "1"}),
                patch("upload_batch.validate_record"),
                patch("upload_batch.HfApi", return_value=api),
                patch("upload_batch.hf_hub_download", side_effect=downloaded_remote),
            ):
                count = merge_and_upload()
            self.assertEqual(15, count)
            api.upload_file.assert_called_once()
            self.assertIn("worker_16", api.upload_file.call_args.kwargs["path_in_repo"])
            self.assertEqual([], list(root.glob("*.jsonl.gz")))
