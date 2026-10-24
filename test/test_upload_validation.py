import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
import upload_batch


class UploadValidationTests(unittest.TestCase):
    def test_rejects_invalid_terminal_game_before_connecting_to_hf(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output_data"
            output.mkdir()
            with gzip.open(output / "bad.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": "bad", "res": 0, "mov": [0]}) + "\n")
            previous = Path.cwd()
            import os
            os.chdir(root)
            try:
                with patch.object(upload_batch, "HfApi") as hf:
                    with self.assertRaisesRegex(RuntimeError, "Refusing invalid"):
                        upload_batch.merge_and_upload()
                    hf.assert_not_called()
            finally:
                os.chdir(previous)
    def test_invalid_identity_never_reaches_hf(self):
        import os
        for bad in ({"id": 99, "ts": 1785000000000},
                    {"id": "good", "ts": -5},
                    {"id": "", "ts": 1785000000000}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "bad.jsonl.gz"
                with gzip.open(target, "wt", encoding="utf-8") as out:
                    out.write(json.dumps(bad) + "\n")
                with (patch.dict(os.environ, {"OUTPUT_DIR": folder, "HF_TOKEN": "fake"}),
                      patch.object(upload_batch, "validate_record"),
                      patch.object(upload_batch, "HfApi") as hf):
                    with self.assertRaisesRegex(RuntimeError, "Refusing invalid"):
                        upload_batch.merge_and_upload()
                    hf.assert_not_called()
                self.assertTrue(target.exists())
