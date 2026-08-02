import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


SCRIPT = Path(__file__).parents[1] / "src" / "workers" / "archive_hf_shard.py"
SPEC = importlib.util.spec_from_file_location("archive_hf_shard", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archive
SPEC.loader.exec_module(archive)


class ArchiveShardTests(unittest.TestCase):
    def test_merge_is_sorted_streaming_and_manifest_matches_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "repo" / "staging" / "worker_3" / "20260802"
            output = root / "output"
            source.mkdir(parents=True)
            output.mkdir()
            for name, lines in (("b.jsonl.gz", ['{"id":2}\n']), ("a.jsonl.gz", ['{"id":1}\n', "\n"])):
                with gzip.open(source / name, "wt", encoding="utf-8") as handle:
                    handle.writelines(lines)

            archive_path, manifest = archive.merge_shard(root / "repo", "worker_3", output)

            with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.readlines(), ['{"id":1}\n', '{"id":2}\n'])
            self.assertEqual(manifest["source_file_count"], 2)
            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(manifest["archive_bytes"], archive_path.stat().st_size)
            self.assertEqual(manifest["sha256"], hashlib.sha256(archive_path.read_bytes()).hexdigest())

    def test_empty_shard_produces_valid_empty_gzip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            archive_path, manifest = archive.merge_shard(root / "repo", "fresh", root / "output")
            with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "")
            self.assertEqual(manifest["source_file_count"], 0)
            self.assertEqual(manifest["record_count"], 0)

    def test_upload_is_one_commit_with_archive_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            archive_path = output / "worker_1.jsonl.gz"
            archive_path.write_bytes(b"archive")
            manifest = {"shard": "worker_1", "sha256": "abc"}
            api = MagicMock()

            archive.upload_archive(api, archive_path, manifest, output)

            api.create_commit.assert_called_once()
            kwargs = api.create_commit.call_args.kwargs
            self.assertEqual(len(kwargs["operations"]), 2)
            self.assertEqual(json.loads((output / "worker_1.manifest.json").read_text()), manifest)

    def test_completed_manifest_only_reuses_the_same_staging_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            archive_dir = repo / "archive"
            archive_dir.mkdir()
            manifest_path = archive_dir / "worker_2.manifest.json"
            manifest = {
                "format": 2,
                "shard": "worker_2",
                "source_staging_tree": "tree-a",
                "sha256": "abc",
                "source_file_count": 10,
                "record_count": 20,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(archive.completed_manifest(repo, "worker_2", "tree-a"), manifest)
            self.assertIsNone(archive.completed_manifest(repo, "worker_2", "tree-b"))


if __name__ == "__main__":
    unittest.main()
