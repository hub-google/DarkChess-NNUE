"""Production must only call shared browser/Node DarkChessEngine for any game."""
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class SharedDecisionEngineProductionTests(unittest.TestCase):
    def test_all_five_waves_use_same_node_engine(self):
        workflow = (ROOT / ".github/workflows/self_play.yml").read_text()
        self.assertEqual(workflow.count("node .shared-build/src/workers/self_play_shared.js &"), 5)
        self.assertNotIn("python -u src/workers/self_play.py &", workflow)
        self.assertEqual(workflow.count("node-version: \"22\""), 5)

    def test_promotion_uses_shared_sprt_not_python_search(self):
        workflow = (ROOT / ".github/workflows/train.yml").read_text()
        self.assertIn("node .shared-build/src/workers/sprt_shared.js", workflow)
        self.assertNotIn("python -u src/training/sprt_validation.py", workflow)
        self.assertIn("steps.validate.outputs.passed == 'true'", workflow)

    def test_one_engine_is_used_by_all_game_entrypoints(self):
        worker = (ROOT / "src/workers/self_play_shared.ts").read_text()
        arena = (ROOT / "src/workers/sprt_shared.ts").read_text()
        browser = (ROOT / "frontend/src/engine/search.worker.ts").read_text()
        for source in (worker, arena, browser):
            self.assertIn("DarkChessEngine", source)
        assert "chooseAiMove" not in (ROOT / "frontend/src/engine/board.ts").read_text()

    def test_self_play_records_actual_completed_search_metrics(self):
        worker = (ROOT / "src/workers/self_play_shared.ts").read_text()
        for required in ("record.qd.push(analysis.depth)", "record.qn.push(analysis.nodes)",
                         "record.qt.push(Math.round(analysis.elapsedMs))",
                         "MIN_AVG_COMPLETED_DEPTH", "MIN_LATE_COMPLETED_DEPTH"):
            self.assertIn(required, worker)

if __name__ == "__main__":
    unittest.main()
