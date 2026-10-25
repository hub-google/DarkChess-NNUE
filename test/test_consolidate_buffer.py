import unittest
from unittest.mock import patch

from src.training.consolidate_buffer import (
    delete_snapshot_files_and_verify,
    select_snapshot_files,
    staging_timestamp,
)


class SnapshotSelectionTests(unittest.TestCase):
    def test_normalizes_legacy_millisecond_timestamps(self):
        self.assertEqual(
            staging_timestamp("staging/fresh/data_1700000000123.jsonl.gz"),
            1700000000,
        )

    def test_selects_only_files_at_or_before_cutoff(self):
        paths = [
            "staging/worker_1/20260803/batch_100.jsonl.gz",
            "staging/worker_2/20260803/batch_200.jsonl.gz",
            "staging/worker_3/20260803/batch_201.jsonl.gz",
        ]
        self.assertEqual(select_snapshot_files(paths, 200), paths[:2])


class SnapshotDeletionTests(unittest.TestCase):
    def test_deletes_snapshot_without_touching_concurrent_upload(self):
        snapshot = {
            "staging/worker_1/20260803/batch_100.jsonl.gz",
            "staging/worker_2/20260803/batch_200.jsonl.gz",
        }
        newer = "staging/worker_3/20260803/batch_300.jsonl.gz"

        class FakeApi:
            def __init__(self):
                self.files = set(snapshot)

            def create_commit(self, **kwargs):
                # Simulate self-play committing a newer batch while cleanup is
                # deleting the immutable snapshot paths.
                self.files.add(newer)
                for operation in kwargs["operations"]:
                    self.files.discard(operation.path_in_repo)

        api = FakeApi()
        with patch(
            "src.training.consolidate_buffer.list_staging_files",
            side_effect=lambda current_api: sorted(current_api.files),
        ):
            delete_snapshot_files_and_verify(api, snapshot)

        self.assertEqual(api.files, {newer})


if __name__ == "__main__":
    unittest.main()
