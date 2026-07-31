import gzip
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKERS_DIR = PROJECT_ROOT / "src" / "workers"
sys.path.insert(0, str(WORKERS_DIR))

import upload_batch  # noqa: E402


class _SuccessfulApi:
    def upload_file(self, **_kwargs):
        return None


class _FailingApi:
    def upload_file(self, **_kwargs):
        raise RuntimeError("simulated upload failure")


class UploadTransactionTests(unittest.TestCase):
    def _write_source(self, root):
        output_dir = Path(root) / "output_data"
        output_dir.mkdir()
        source = output_dir / "source.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write('{"id":"game-1"}\n')
        return source

    def test_sources_are_deleted_only_after_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            previous = os.getcwd()
            os.chdir(temp_dir)
            try:
                with patch.object(upload_batch, "HfApi", return_value=_SuccessfulApi()):
                    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
                        upload_batch.merge_and_upload()
            finally:
                os.chdir(previous)
            self.assertFalse(source.exists())

    def test_sources_survive_upload_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            previous = os.getcwd()
            os.chdir(temp_dir)
            try:
                with patch.object(upload_batch, "HfApi", return_value=_FailingApi()):
                    with patch.object(upload_batch.time, "sleep", return_value=None):
                        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
                            with self.assertRaisesRegex(RuntimeError, "simulated"):
                                upload_batch.merge_and_upload()
            finally:
                os.chdir(previous)
            self.assertTrue(source.exists())
            self.assertEqual(list((Path(temp_dir) / "output_data").glob("*.jsonl.gz")), [source])


if __name__ == "__main__":
    unittest.main()
