"""Contracts for the single consolidated self-play workflow and training.

The retired self_play_wave.yml and publish_training_status.yml files are
intentionally absent: only three active workflows remain.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class AutonomousWorkflowContractTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_one_run_fifteen_workers_for_five_hours_then_one_hf_writer(self):
        code = self.read("self_play.yml")
        self.assertEqual(code.count("  selfplay-workers:"), 1)
        self.assertEqual(code.count("  upload:"), 1)
        self.assertEqual(code.count("  continue-next-run:"), 1)
        self.assertNotIn("  wave2-workers:", code)
        self.assertIn("needs: selfplay-workers", code)
        self.assertIn("needs: [upload, selfplay-workers]", code)
        self.assertIn("needs.upload.result == 'success'", code)
        self.assertEqual(code.count("worker_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"), 1)
        self.assertEqual(code.count("max-parallel: 15"), 1)
        self.assertEqual(code.count("BUDGET=18000"), 1)
        self.assertEqual(code.count("timeout-minutes: 345"), 1)
        self.assertEqual(code.count("actions/upload-artifact@v4"), 2)
        self.assertEqual(code.count("actions/download-artifact@v4"), 1)
        self.assertEqual(code.count("merge-multiple: true"), 1)
        self.assertEqual(code.count("WORKER_ID: \"16\""), 1)
        self.assertEqual(code.count("python -u src/workers/upload_batch.py"), 1)
        self.assertIn("group: darkchess-hf-stage-writer", code)
        self.assertIn("if: ${{ always() && !cancelled() }}", code)

    def test_daily_train_keeps_sprt(self):
        code = self.read("train.yml")
        self.assertIn("replay_gate.py", code)
        self.assertIn("steps.validate.outputs.passed == 'true'", code)
        self.assertIn('cron: "41 3 * * *"', code)

    def test_watchdog_recovers_without_undoing_manual_cancel(self):
        code = self.read("self_play.yml")
        self.assertIn("recovery-watchdog:", code)
        self.assertIn("actions/workflows/self_play.yml/runs", code)
        self.assertIn('status != "completed"', code)
        self.assertIn('LATEST_RESULT" = "cancelled"', code)
        self.assertIn("gh workflow run self_play.yml", code)

    def test_training_handoff_after_single_upload(self):
        code = self.read("self_play.yml")
        self.assertIn("continue-next-run:", code)
        self.assertIn("needs.upload.result == 'success'", code)
        self.assertIn("gh workflow run train.yml", code)
        self.assertIn("第16號棋譜上傳員", code)
        self.assertIn("No complete new games; saved checkpoints", code)
        self.assertIn("CURRENT_STATUS", code)
        self.assertIn("LAST_TRAIN=", code)
        self.assertIn("21600", code)

    def test_monthly_actions_keepalive(self):
        code = self.read("self_play.yml")
        self.assertIn('cron: "43 3 1 * *"', code)
        self.assertIn("git commit --allow-empty", code)
        self.assertIn("每月維持自動排程運作", code)

    def test_failed_cpu_worker_cannot_suppress_single_writer(self):
        code = self.read("self_play.yml")
        self.assertIn("group: self-play-v4-", code)
        self.assertIn("fail-fast: false", code)
        self.assertIn("if: ${{ always() && !cancelled() && github.event_name == 'workflow_dispatch' }}", code)
        self.assertIn("trap stop_child INT TERM", code)
        self.assertIn("selfplay-worker-", code)
        self.assertIn("resume_state/*.json", code)
        self.assertIn("resume_state/*.memo", code)

    def test_hf_upload_is_in_same_run_with_no_periodic_repair_job(self):
        code = self.read("self_play.yml")
        self.assertNotIn("  maintenance:", code)
        self.assertNotIn('cron: "17 */3 * * *"', code)
        self.assertNotIn("recover_failed_uploads.py", code)
        self.assertIn("HF upload or remote read-back failed", code)
        self.assertIn('LAST_RESULT" != "success"', code)
        self.assertIn("python -u src/workers/upload_batch.py", code)

    def test_only_three_active_workflows(self):
        self.assertEqual(
            {p.name for p in ROOT.glob("*.yml")},
            {"self_play.yml", "train.yml", "deploy_pages.yml"},
        )


if __name__ == "__main__":
    unittest.main()
