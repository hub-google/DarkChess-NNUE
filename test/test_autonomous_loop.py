import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"

class AutonomousWorkflowContractTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / name).read_text(encoding="utf-8")

    def test_five_sequential_waves(self):
        code = self.read("self_play.yml")
        for i in range(1, 6):
            self.assertIn(f"  wave{i}:", code)
            self.assertIn(f"wave: {i}", code)
            if i > 1:
                self.assertIn(f"needs: wave{i-1}", code)
        self.assertIn("needs: wave5", code)
        self.assertIn("gh workflow run self_play.yml", code)
        self.assertIn('cron: "23 * * * *"', code)

    def test_fifteen_cpu_one_hf_writer_and_artifacts(self):
        code = self.read("self_play_wave.yml")
        self.assertIn("workflow_call:", code)
        self.assertIn("worker_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]", code)
        self.assertIn("BUDGET=3000", code)
        self.assertIn("timeout-minutes: 58", code)
        self.assertIn("actions/upload-artifact@v4", code)
        self.assertIn("actions/download-artifact@v4", code)
        self.assertIn("merge-multiple: true", code)
        self.assertIn("hf-writer:", code)
        self.assertIn("needs: workers", code)
        self.assertIn("group: darkchess-hf-stage-writer", code)
        self.assertIn('WORKER_ID: "16"', code)
        self.assertEqual(1, code.count("HF_TOKEN:"))
        self.assertIn("python -u src/workers/upload_batch.py", code)

    def test_daily_train_keeps_sprt(self):
        code = self.read("train.yml")
        self.assertIn("replay_gate.py", code)
        self.assertIn("steps.validate.outputs.passed == 'true'", code)
        self.assertIn('cron: "41 3 * * *"', code)

    def test_watchdog_recovers_stalled_self_play(self):
        code = self.read("self_play.yml")
        self.assertIn("recovery-watchdog:", code)
        self.assertIn('cron: "23 * * * *"', code)
        self.assertIn("actions/workflows/self_play.yml/runs", code)
        self.assertIn('status != "completed"', code)
        self.assertIn("gh workflow run self_play.yml", code)

    def test_training_handoff_after_fifth_wave(self):
        code = self.read("self_play.yml")
        self.assertIn("continue-next-cycle:", code)
        self.assertIn("needs: wave5", code)
        self.assertIn("needs.wave5.result == 'success'", code)
        self.assertIn("gh workflow run train.yml", code)
        self.assertIn("LAST_TRAIN=", code)
        self.assertIn("21600", code)

    def test_monthly_actions_keepalive(self):
        code = self.read("self_play.yml")
        self.assertIn('cron: "43 3 1 * *"', code)
        self.assertIn("git commit --allow-empty", code)
        self.assertIn("Keep scheduled Actions active with an empty commit", code)
