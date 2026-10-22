"""Static workflow contract tests: block accidental disabling of the AI loop."""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / ".github" / "workflows"


class AutonomousWorkflowContractTests(unittest.TestCase):
    def read(self, workflow):
        return (ROOT / workflow).read_text(encoding="utf-8")

    def test_scheduled_generation_is_bounded(self):
        code = self.read("self_play.yml")
        self.assertIn('cron: "17 */6 * * *"', code)
        self.assertIn("GENERATION_WINDOW_SECONDS", code)
        self.assertIn("'3300'", code)
        self.assertIn("fromJSON(github.event_name == 'schedule'", code)

    def test_completion_bridge_starts_real_training(self):
        code = self.read("train_after_self_play.yml")
        self.assertIn("workflow_run:", code)
        self.assertIn('"Distributed belief-state self-play"', code)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", code)
        self.assertIn("gh workflow run train.yml", code)

    def test_private_dataset_gate_before_training(self):
        code = self.read("train.yml")
        self.assertIn("token=token", code)
        self.assertIn("replay_gate.py", code)
        self.assertIn("steps.enough_games.outputs.ready == 'true'", code)
        self.assertIn("steps.validate.outputs.passed == 'true'", code)

    def test_monthly_empty_commit_keepalive(self):
        code = self.read("monthly_activity.yml")
        self.assertIn('cron: "43 3 1 * *"', code)
        self.assertIn("git commit --allow-empty", code)
