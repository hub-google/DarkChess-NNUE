"""Mocked safety checks for irreversible HF history cleanup."""
import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

SCRIPT = Path(__file__).resolve().parents[1] / "src/workers/compact_hf_history.py"
spec = importlib.util.spec_from_file_location("compact_hf_history", SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def file(path, size=100, blob="aaa", lfs=None):
    return types.SimpleNamespace(path=path, size=size, blob_id=blob, lfs=lfs)


class HistoryCleanupTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.files = [
            file("replay_buffer.jsonl.gz", 100),
            file("staging/worker_16/run.gz", 50),
            file("receipts/backfill/verified.json", 10),
        ]
        self.api.list_repo_tree.return_value = self.files
        self.api.list_repo_commits.side_effect = [[1, 2], [3]]

    def test_squash_keeps_replay_staging_and_receipts(self):
        cleanup.compact_history(self.api, attempts=1)
        self.api.super_squash_history.assert_called_once_with(
            repo_id=cleanup.REPO_ID,
            repo_type="dataset",
            branch="main",
            commit_message="保留目前棋譜，清除已失效的 HF 歷史版本",
        )

    def test_stop_when_any_current_file_disappears(self):
        self.api.list_repo_tree.side_effect = [self.files, self.files[:-1]]
        with self.assertRaisesRegex(RuntimeError, "Live dataset tree changed"):
            cleanup.compact_history(self.api, attempts=1)

    def test_stop_when_replay_and_staging_missing(self):
        self.api.list_repo_tree.return_value = [file("receipts/proof.json")]
        with self.assertRaisesRegex(RuntimeError, "No active replay"):
            cleanup.compact_history(self.api, attempts=1)
        self.api.super_squash_history.assert_not_called()

    def test_no_unnecessary_squash_when_already_one_commit(self):
        self.api.list_repo_commits.side_effect = [[1], [1]]
        cleanup.compact_history(self.api, attempts=1)
        self.api.super_squash_history.assert_not_called()

    def test_retry_when_server_temporarily_rejects_squash(self):
        self.api.super_squash_history.side_effect = [OSError("busy"), None]
        self.api.list_repo_commits.side_effect = [[1, 2], [1, 2], [3]]
        cleanup.compact_history(self.api, attempts=2, sleep=lambda _: None)
        self.assertEqual(self.api.super_squash_history.call_count, 2)

    def test_fail_when_history_still_has_many_commits(self):
        self.api.list_repo_commits.side_effect = [[1, 2], [1, 2]]
        with self.assertRaisesRegex(RuntimeError, "Expected 1 HF commit"):
            cleanup.compact_history(self.api, attempts=1)


if __name__ == "__main__":
    unittest.main()
