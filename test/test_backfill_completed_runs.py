"""Backfill must upload genuine nested Actions worker artifacts, never checkpoints."""
import gzip
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
import backfill_completed_runs as backfill


class BackfillTests(unittest.TestCase):
    def test_nested_worker_artifact_is_uploaded_and_verified(self):
        api = MagicMock()
        api.file_exists.return_value = False
        seen = []

        def fake_gh(*args):
            if args[0] == "api":
                return json.dumps({"artifacts": [
                    {"id": 777, "name": "selfplay-worker-1-123", "expired": False}]})
            self.assertEqual(args[:3], ("run", "download", 123))
            destination = Path(args[args.index("--dir") + 1])
            source = destination / "output_data" / "game.jsonl.gz"
            source.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(source, "wt", encoding="utf-8") as fp:
                fp.write('{"id":"one"}\\n')
            return ""

        def fake_merge():
            output = Path(os.environ["OUTPUT_DIR"])
            found = list(output.glob("*.jsonl.gz"))
            seen.extend(found)
            self.assertEqual(len(found), 1)
            with gzip.open(found[0], "rt", encoding="utf-8") as fp:
                self.assertIn('"id":"one"', fp.read())
            return 1

        def fake_download(**kwargs):
            result = Path(kwargs["local_dir"]) / "receipt.json"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_bytes(Path(api.upload_file.call_args.kwargs["path_or_fileobj"]).read_bytes())
            return str(result)

        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                     "GH_TOKEN": "test", "HF_TOKEN": "test"}), (
             patch.object(backfill, "HfApi", return_value=api)), (
             patch.object(backfill, "gh", side_effect=fake_gh)), (
             patch.object(backfill, "merge_and_upload", side_effect=fake_merge)), (
             patch.object(backfill, "hf_hub_download", side_effect=fake_download)):
            backfill.backfill([123])
        self.assertEqual(len(seen), 1)
        self.assertEqual(api.upload_file.call_args.kwargs["path_in_repo"], "receipts/backfill/123.json")

    def test_checkpoint_only_round_does_not_claim_upload(self):
        api = MagicMock()
        api.file_exists.return_value = False
        def fake_gh(*args):
            if args[0] == "api":
                return json.dumps({"artifacts": [
                    {"id": 77, "name": "selfplay-worker-1-123", "expired": False}]})
            folder = Path(args[args.index("--dir") + 1]) / "resume_state"
            folder.mkdir(parents=True)
            (folder / "worker_1.json").write_text("{}")
            return ""
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                     "GH_TOKEN": "test", "HF_TOKEN": "test"}), (
             patch.object(backfill, "HfApi", return_value=api)), (
             patch.object(backfill, "gh", side_effect=fake_gh)), (
             patch.object(backfill, "merge_and_upload") as uploader):
            backfill.backfill([123])
        uploader.assert_not_called()
        api.upload_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
