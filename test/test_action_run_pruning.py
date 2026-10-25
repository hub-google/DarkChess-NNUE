"""Dry-run safety checks for GitHub Actions run pruning (no API writes)."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))
import prune_action_runs as pruner


def run(id, event="workflow_dispatch", status="completed", title="", created="2026-10-25"):
    return {"id": id, "event": event, "status": status, "created_at": created,
            "display_title": title, "conclusion": "success"}


class ActionRunPruningTests(unittest.TestCase):
    def test_separate_watchdog_from_hf_maintenance_and_real_selfplay(self):
        cases = [
            (run(1), "self-play"),
            (run(2, "schedule", title=pruner.WATCHDOG_TITLE), "watchdog"),
            (run(3, "schedule", title=pruner.MAINTENANCE_TITLE), "HF maintenance"),
            (run(4, "schedule", title=pruner.KEEPALIVE_TITLE), "keepalive"),
        ]
        for item, expected in cases:
            self.assertEqual(pruner.run_group("self_play", item), expected)

    def test_keep_newest_even_if_active_and_never_delete_active(self):
        runs = [run(10, created="2026-10-24"), run(11, created="2026-10-25", status="in_progress")]
        result = pruner.choose_deletions("self_play", runs, protected_run_id=11)
        self.assertEqual([r["id"] for _, r in result], [10])
        self.assertEqual(pruner.choose_deletions("self_play",
                         [run(1, status="in_progress", created="2026-10-24"),
                          run(2, created="2026-10-25")]), [])

    def test_cancel_is_last_manual_round_and_not_erased_by_watchdog(self):
        runs = [
            run(21, status="completed", created="2026-10-24"),
            run(22, status="completed", created="2026-10-25"),
            run(23, event="schedule", title=pruner.WATCHDOG_TITLE,
                created="2026-10-26"),
        ]
        runs[1]["conclusion"] = "cancelled"
        selected = pruner.choose_deletions("self_play", runs)
        self.assertEqual([x["id"] for _, x in selected], [21])

    def test_old_completed_ci_runs_pruned_separately_from_production(self):
        runs = [
            run(1, event="push", title="test old", created="2026-10-23"),
            run(2, event="push", title="test newest", created="2026-10-25"),
            run(3, event="workflow_dispatch", created="2026-10-24"),
        ]
        self.assertEqual([(g, r["id"]) for g, r in pruner.choose_deletions("train", runs)],
                         [("python regression", 1)])

    def test_hf_writer_verified_means_worker_artifacts_can_be_deleted(self):
        artifacts = [{"name": "selfplay-wave-3-worker-2-55", "id": 3, "expired": False}]
        jobs = [{"name": "Wave 3 / worker 16 HF writer", "conclusion": "success"}]
        with patch.object(pruner, "pages", side_effect=[
            artifacts, jobs,
        ]):
            self.assertTrue(pruner.self_play_artifacts_safe(55))

    def test_failed_writer_requires_verified_hf_receipt(self):
        artifacts = [
            {"name": "selfplay-wave-3-worker-2-55", "id": 3, "expired": False},
            {"name": "hf-upload-recovery-wave-3-55", "id": 4, "expired": False},
        ]
        jobs = [{"name": "Wave 3 / worker 16 HF writer", "conclusion": "failure"}]
        with patch.object(pruner, "pages", side_effect=[artifacts, jobs]):
            self.assertFalse(pruner.self_play_artifacts_safe(55, hf_api=None))
        api = MagicMock()
        api.file_exists.return_value = False
        with patch.object(pruner, "pages", side_effect=[artifacts, jobs]):
            self.assertFalse(pruner.self_play_artifacts_safe(55, api))
        api.file_exists.return_value = True
        with patch.object(pruner, "pages", side_effect=[artifacts, jobs]):
            self.assertTrue(pruner.self_play_artifacts_safe(55, api))
        api.file_exists.assert_called()

    def test_legacy_writer_name_is_supported(self):
        jobs = [{"name": "wave1 / Wave 1 / worker 16 HF writer", "conclusion": "success"}]
        with patch.object(pruner, "pages", return_value=jobs):
            self.assertEqual(pruner._writer_waves(45), {"1"})

    def test_refuses_to_delete_live_runs_even_if_stale(self):
        calls = []
        def fake_api(resource, method="GET"):
            calls.append((resource, method))
            if "/workflows/" in resource:
                return {"workflow_runs": [
                    run(8, created="2026-10-24"), run(9, created="2026-10-25")]}
            if resource == "actions/runs/8":
                return {"status": "in_progress"}
            raise AssertionError(resource)
        with patch.dict(os.environ, {"GH_TOKEN": "fake", "GITHUB_RUN_ID": "9",
                                      "HF_TOKEN": ""}), patch.object(pruner, "gh_api", side_effect=fake_api):
            result = pruner.prune(scope="deploy")
        self.assertEqual(result, (0, 1))
        self.assertFalse(any(method == "DELETE" for _, method in calls))


if __name__ == "__main__":
    unittest.main()
