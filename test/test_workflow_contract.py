"""Regression contract for the three active GitHub Actions workflows.

GitHub decides the exact termination timing of the Cancel button. These guards
prevent downstream salvage jobs and new wave dispatches after cancellation.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class WorkflowArchitectureTests(unittest.TestCase):
    def test_only_three_active_workflow_files(self):
        self.assertEqual(
            {p.name for p in WORKFLOWS.glob("*.yml")},
            {"self_play.yml", "train.yml", "deploy_pages.yml"},
        )

    def test_five_waves_with_fifteen_workers_and_one_writer(self):
        content = (WORKFLOWS / "self_play.yml").read_text(encoding="utf-8")
        for wave in range(1, 6):
            self.assertIn(f"  wave{wave}-workers:", content)
            self.assertIn(f"  wave{wave}-upload:", content)
        self.assertIn("max-parallel: 15", content)
        self.assertEqual(content.count("第16號棋譜上傳員"), 5)
        self.assertIn("self-play-v3", content)
        self.assertIn("SEARCH_DEPTH=", content)
        self.assertEqual(content.count("SEARCH_DEPTH="), 5)
        self.assertIn("Production paused", content)

    def test_every_always_condition_checks_cancellation(self):
        for file in WORKFLOWS.glob("*.yml"):
            lines = file.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, 1):
                if line.strip().startswith("if:") and "always()" in line:
                    self.assertIn("!cancelled()", line, (file.name, lineno))

    def test_watchdog_respects_manual_cancel(self):
        script = (WORKFLOWS / "self_play.yml").read_text(encoding="utf-8")
        self.assertIn('LATEST_RESULT" = "cancelled"', script)
        self.assertIn('CURRENT_STATUS" != "in_progress"', script)
        self.assertIn("github.event.schedule == '17 */3 * * *'", script)


if __name__ == "__main__":
    unittest.main()
