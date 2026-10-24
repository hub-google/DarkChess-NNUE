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

    def test_five_sequential_waves(self):
        code = self.read("self_play.yml")
        for wave in range(1, 6):
            self.assertIn(f"  wave{wave}-workers:", code)
            self.assertIn(f"  wave{wave}-upload:", code)
            self.assertIn(f"needs: wave{wave}-workers", code)
            if wave > 1:
                self.assertIn(f"needs: wave{wave-1}-upload", code)
        self.assertIn("needs: wave5-upload", code)
        self.assertIn("gh workflow run self_play.yml", code)
        self.assertIn('cron: "23 * * * *"', code)

    def test_fifteen_cpu_one_hf_writer_per_wave_and_artifacts(self):
        code = self.read("self_play.yml")
        self.assertEqual(code.count("worker_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"), 5)
        self.assertEqual(code.count("max-parallel: 15"), 5)
        self.assertEqual(code.count("BUDGET=3000"), 5)
        self.assertEqual(code.count("timeout-minutes: 58"), 6)  # 5 workers + maintenance
        self.assertEqual(code.count("actions/upload-artifact@v4"), 10)
        self.assertEqual(code.count("actions/download-artifact@v4"), 5)
        self.assertEqual(code.count("merge-multiple: true"), 5)
        self.assertEqual(code.count("name: Wave "), 10)
        for wave in range(1, 6):
            writer = code.split(f"  wave{wave}-upload:")[1].split("\n  ", 1)[0]
            self.assertIn("HF", writer)
        self.assertIn("group: darkchess-hf-stage-writer", code)
        self.assertEqual(code.count('WORKER_ID: "16"'), 5)
        self.assertEqual(code.count("python -u src/workers/upload_batch.py"), 5)
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

    def test_training_handoff_after_fifth_wave(self):
        code = self.read("self_play.yml")
        self.assertIn("continue-next-cycle:", code)
        self.assertIn("needs: wave5-upload", code)
        self.assertIn("always() && !cancelled()", code)
        self.assertIn("gh workflow run train.yml", code)
        self.assertIn("worker 16 HF writer", code)
        self.assertIn("No verified upload waves", code)
        self.assertIn("CURRENT_STATUS", code)
        self.assertIn("LAST_TRAIN=", code)
        self.assertIn("21600", code)

    def test_monthly_actions_keepalive(self):
        code = self.read("self_play.yml")
        self.assertIn('cron: "43 3 1 * *"', code)
        self.assertIn("git commit --allow-empty", code)
        self.assertIn("Keep scheduled Actions alive once a month", code)

    def test_failed_cpu_worker_does_not_block_subsequent_waves(self):
        code = self.read("self_play.yml")
        self.assertIn("group: self-play-v3-", code)
        for wave in range(2, 6):
            workers = code.split(f"  wave{wave}-workers:")[1].split("\n  wave", 1)[0]
            self.assertIn("always()", workers)
            self.assertIn("!cancelled()", workers)
        self.assertIn("trap stop_child INT TERM", code)

    def test_training_status_is_published_and_pages_redeployed(self):
        code = self.read("self_play.yml")
        self.assertIn("maintenance:", code)
        self.assertIn('cron: "17 */3 * * *"', code)
        self.assertIn("publish_training_status.py", code)
        self.assertIn("recover_failed_uploads.py", code)
        self.assertIn("gh workflow run deploy_pages.yml", code)
        self.assertIn("contents: write", code)

    def test_only_three_active_workflows(self):
        self.assertEqual(
            {p.name for p in ROOT.glob("*.yml")},
            {"self_play.yml", "train.yml", "deploy_pages.yml"},
        )


if __name__ == "__main__":
    unittest.main()
