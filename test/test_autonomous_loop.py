"""Static workflow contract tests: block accidental disabling of the AI loop."""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class AutonomousWorkflowContractTests(unittest.TestCase):
    def read(self, workflow):
        return (ROOT / workflow).read_text(encoding="utf-8")

    def test_continuous_generation_is_bounded_and_self_restarts(self):
        code = self.read("self_play.yml")
        self.assertIn("inputs.continuous", code)
        self.assertIn("GENERATION_WINDOW_SECONDS", code)
        self.assertIn('GENERATION_WINDOW_SECONDS: "20400"', code)
        self.assertIn('timeout-minutes: 355', code)
        self.assertNotIn("'3300'", code)
        self.assertIn("continue-next-cycle:", code)
        self.assertIn("needs.self-play-worker.result == 'success'", code)
        self.assertIn("gh workflow run self_play.yml", code)
        self.assertIn("-f continuous=true", code)
        self.assertIn("UPLOADED_BATCHES", code)
        self.assertIn('compgen -G "output_data/*.jsonl.gz"', code)

    def test_hourly_watchdog_recovers_idle_or_failed_loop(self):
        code = self.read("self_play_watchdog.yml")
        self.assertIn('cron: "23 * * * *"', code)
        self.assertIn("workflow_dispatch:", code)
        self.assertIn("gh workflow run self_play.yml", code)
        self.assertIn("actions/workflows/self_play.yml/runs", code)
        self.assertIn('status != "completed"', code)

    def test_completion_bridge_starts_real_training(self):
        code = self.read("train_after_self_play.yml")
        self.assertIn("workflow_run:", code)
        self.assertIn('"Distributed belief-state self-play"', code)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", code)
        self.assertIn("gh workflow run train.yml", code)
        self.assertIn("21600", code)
        self.assertNotIn("workflow_run.event != 'schedule'", code)

    def test_private_dataset_gate_before_training(self):
        code = self.read("train.yml")
        self.assertIn("token=token", code)
        self.assertIn("replay_gate.py", code)
        self.assertIn("steps.enough_games.outputs.ready == 'true'", code)
        self.assertIn('cron: "41 3 * * *"', code)
        self.assertIn("steps.validate.outputs.passed == 'true'", code)

    def test_monthly_empty_commit_keepalive(self):
        code = self.read("monthly_activity.yml")
        self.assertIn('cron: "43 3 1 * *"', code)
        self.assertIn("git commit --allow-empty", code)
