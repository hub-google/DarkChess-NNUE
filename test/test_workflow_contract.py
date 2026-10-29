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

    def test_single_run_with_fifteen_workers_and_one_writer(self):
        content = (WORKFLOWS / "self_play.yml").read_text(encoding="utf-8")
        self.assertEqual(content.count("  selfplay-workers:"), 1)
        self.assertEqual(content.count("  upload:"), 1)
        self.assertEqual(content.count("  continue-next-run:"), 1)
        self.assertEqual(content.count("max-parallel: 15"), 1)
        self.assertEqual(content.count("第16號棋譜上傳員"), 1)
        self.assertIn("self-play-v4", content)
        self.assertEqual(content.count("SEARCH_DEPTH="), 1)
        self.assertIn("BUDGET=18000", content)
        self.assertIn("needs.upload.result == 'success'", content)
        self.assertIn("needs.selfplay-workers.result == 'success'", content)
        self.assertIn("Production paused", content)

    def test_worker16_extracts_artifacts_at_repository_root(self):
        content = (WORKFLOWS / "self_play.yml").read_text(encoding="utf-8")
        self.assertIn("pattern: selfplay-worker-*-${{ github.run_id }}", content)
        downloader = content.split("- uses: actions/download-artifact@v4", 1)[1].split("- name: 列出第16號實際收到的完整棋譜", 1)[0]
        self.assertIn("          path: .", downloader)
        self.assertNotIn("          path: output_data", downloader)
        self.assertIn("find output_data -maxdepth 1", content)
        self.assertIn("find output_data -mindepth 2", content)
        self.assertIn('CHECKPOINTS" -ne 15', content)
        self.assertIn("VERIFIED all 15 worker Artifacts", content)
        self.assertIn("HF upload or remote read-back failed", content)

    def test_every_always_condition_checks_cancellation(self):
        for file in WORKFLOWS.glob("*.yml"):
            lines = file.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, 1):
                if line.strip().startswith("if:") and "always()" in line:
                    self.assertIn("!cancelled()", line, (file.name, lineno))

    def test_switch_to_new_run_cancels_old_before_dispatch(self):
        script = (WORKFLOWS / "train.yml").read_text(encoding="utf-8")
        self.assertIn("[切換單輪]", script)
        self.assertIn("actions/runs/$OLD_RUN/cancel", script)
        self.assertIn("prevent double CPU consumption", script)

    def test_watchdog_respects_manual_cancel(self):
        script = (WORKFLOWS / "self_play.yml").read_text(encoding="utf-8")
        self.assertIn('LATEST_RESULT" = "cancelled"', script)
        self.assertIn('CURRENT_STATUS" != "in_progress"', script)
        self.assertIn('LAST_RESULT" != "success"', script)
        self.assertNotIn('cron: "17 */3 * * *"', script)
        self.assertNotIn("recover_failed_uploads.py", script)
        self.assertNotIn("  maintenance:", script)


if __name__ == "__main__":
    unittest.main()
