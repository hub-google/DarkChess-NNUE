"""Offline checks for at-least-once HF recovery with exactly-once receipts."""
import gzip
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
import recover_failed_uploads as recovery


def api_result(resource):
    if "/workflows/self_play.yml/runs?" in resource:
        return {"workflow_runs": [
            {"id": 123, "status": "completed", "conclusion": "failure", "created_at": "2099-01-01T00:00:00Z"}
        ]}
    if "/runs/123/artifacts?" in resource:
        return {"artifacts": [
            {"id": 999, "expired": False, "name": "hf-upload-recovery-wave-3-123"}
        ]}
    raise AssertionError(f"Unexpected GitHub API call: {resource}")


class RecoveryTests(unittest.TestCase):
    def test_verified_receipt_prevents_reupload(self):
        api = MagicMock()
        api.file_exists.return_value = True
        with (
            patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                    "GH_TOKEN": "fake", "HF_TOKEN": "fake"}),
            patch.object(recovery, "HfApi", return_value=api),
            patch.object(recovery, "gh_json", side_effect=api_result),
            patch.object(recovery, "merge_and_upload") as upload,
        ):
            recovery.recover()
        upload.assert_not_called()
        api.upload_file.assert_not_called()

    def test_successful_overall_run_still_recovers_failed_wave(self):
        api = MagicMock()
        api.file_exists.return_value = True
        def fake_api(resource):
            if "/workflows/self_play.yml/runs?" in resource:
                return {"workflow_runs": [
                    {"id": 123, "status": "completed", "conclusion": "success",
                     "created_at": "2099-01-01T00:00:00Z"}]}
            if "/runs/123/artifacts?" in resource:
                return {"artifacts": [{"id": 88, "expired": False,
                                       "name": "selfplay-wave-3-worker-4-123"}]}
            if "/runs/123/jobs?" in resource:
                return {"jobs": [
                    {"name": "Wave 3 / worker 16 HF writer", "conclusion": "failure"},
                    {"name": "Wave 4 / worker 16 HF writer", "conclusion": "success"}]}
            raise AssertionError(resource)
        with (patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                      "GH_TOKEN": "fake", "HF_TOKEN": "fake"}),
              patch.object(recovery, "HfApi", return_value=api),
              patch.object(recovery, "gh_json", side_effect=fake_api),
              patch.object(recovery, "merge_and_upload") as upload):
            recovery.recover()
        upload.assert_not_called()
        api.file_exists.assert_called_once()
        self.assertEqual(
            "receipts/recovery/123/wave-3.json",
            api.file_exists.call_args.kwargs["filename"],
        )

    def test_new_single_run_writer_success_does_not_reupload(self):
        api = MagicMock()
        api.file_exists.return_value = False
        def new_api(resource):
            if "/workflows/self_play.yml/runs?" in resource:
                return {"workflow_runs": [{"id": 123, "status": "completed",
                                           "conclusion": "success",
                                           "created_at": "2099-01-01T00:00:00Z"}]}
            if "/runs/123/artifacts?" in resource:
                return {"artifacts": [{"id": 88, "expired": False,
                                       "name": "selfplay-worker-4-123"}]}
            if "/runs/123/jobs?" in resource:
                return {"jobs": [{"name": "第16號棋譜上傳員", "conclusion": "success"}]}
            raise AssertionError(resource)
        with (patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                      "GH_TOKEN": "fake", "HF_TOKEN": "fake"}),
              patch.object(recovery, "HfApi", return_value=api),
              patch.object(recovery, "gh_json", side_effect=new_api),
              patch.object(recovery, "merge_and_upload") as upload):
            recovery.recover()
        upload.assert_not_called()
        api.file_exists.assert_not_called()

    def test_single_run_recovery_artifact_is_discoverable(self):
        api = MagicMock()
        api.file_exists.return_value = True
        def new_api(resource):
            if "/workflows/self_play.yml/runs?" in resource:
                return {"workflow_runs": [{"id": 123, "status": "completed",
                                           "conclusion": "failure",
                                           "created_at": "2099-01-01T00:00:00Z"}]}
            if "/runs/123/artifacts?" in resource:
                return {"artifacts": [{"id": 55, "expired": False,
                                       "name": "hf-upload-recovery-run-123"}]}
            raise AssertionError(resource)
        with (patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                      "GH_TOKEN": "fake", "HF_TOKEN": "fake"}),
              patch.object(recovery, "HfApi", return_value=api),
              patch.object(recovery, "gh_json", side_effect=new_api),
              patch.object(recovery, "merge_and_upload") as upload):
            recovery.recover()
        upload.assert_not_called()
        self.assertEqual("receipts/recovery/123/run.json",
                         api.file_exists.call_args.kwargs["filename"])

    def test_failed_artifact_reuploaded_and_acknowledged(self):
        api = MagicMock()
        api.file_exists.return_value = False
        def fake_download(cmd, **kwargs):
            destination = Path(cmd[cmd.index("--dir") + 1])
            with gzip.open(destination / "replay.jsonl.gz", "wt", encoding="utf-8") as file:
                file.write(json.dumps({"id": "game-1", "ts": 1785000000000}) + "\n")
            return MagicMock(returncode=0)
        def fake_receipt(**kwargs):
            return str(Path(kwargs["local_dir"]).parent / "receipt.json")
        with (
            patch.dict(os.environ, {"GITHUB_REPOSITORY": "hub-google/DarkChess-NNUE",
                                    "GH_TOKEN": "fake", "HF_TOKEN": "fake"}),
            patch.object(recovery, "HfApi", return_value=api),
            patch.object(recovery, "gh_json", side_effect=api_result),
            patch.object(recovery.subprocess, "run", side_effect=fake_download),
            patch.object(recovery, "merge_and_upload", return_value=1) as upload,
            patch.object(recovery, "hf_hub_download", side_effect=fake_receipt),
        ):
            recovery.recover()
        upload.assert_called_once()
        api.upload_file.assert_called_once()
        self.assertEqual("receipts/recovery/123/wave-3.json",
                         api.upload_file.call_args.kwargs["path_in_repo"])


if __name__ == "__main__":
    unittest.main()
