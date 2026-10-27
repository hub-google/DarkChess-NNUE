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

    def test_complete_depth_without_node_or_time_limits(self):
        worker = (ROOT / "src/workers/self_play_shared.ts").read_text()
        engine = (ROOT / "frontend/src/engine/search.ts").read_text()
        workflow = (ROOT / ".github/workflows/self_play.yml").read_text()
        self.assertIn("if (ply === 0) return 0", worker)
        self.assertIn("if (ply < 10) return 1", worker)
        self.assertIn("return depth // Move 11 onward", worker)
        self.assertIn("SELF_PLAY_TARGET_DEPTH", worker)
        self.assertIn("Production paused until one CPU pilot completes a validated game.", workflow)
        self.assertIn("const OPENING_SQUARES = [10, 13, 18, 21]", worker)
        self.assertIn("depthAllowed(d, ply)", worker)
        self.assertIn("depth === 0 ? fixedOpening(board)", worker)
        self.assertIn("strategy: SEARCH_STRATEGY", worker)
        self.assertIn("SELF_PLAY_VALIDATE_OPENING", worker)
        self.assertIn("const analysis = fixedOpening(board)", worker)
        self.assertIn("requireFullDepth: true", worker)
        self.assertIn("Complete-depth search must not have a node or time cutoff", engine)
        self.assertNotIn("SEARCH_MAX_SECONDS:", workflow)
        self.assertNotIn("SEARCH_NODE_BUDGET_LATE:", workflow)
        self.assertEqual(workflow.count("BUDGET=20400"), 5)
        self.assertEqual(workflow.count("timeout-minutes: 345"), 5)

    def test_resumes_saved_complete_root_branches_between_waves(self):
        worker = (ROOT / "src/workers/self_play_shared.ts").read_text()
        workflow = (ROOT / ".github/workflows/self_play.yml").read_text()
        for required in ("saveCheckpoint", "restoreCheckpoint", "onRootCompleted:",
                         "resumeRootValues: rootValues", "analysis.depth !== depth"):
            self.assertIn(required, worker)
        self.assertEqual(workflow.count("接續前輪尚未完成的深度搜尋"), 5)
        self.assertEqual(workflow.count("resume_state/*.json"), 5)
        self.assertEqual(workflow.count("resume_state/*.memo"), 5)
        self.assertEqual(workflow.count("validated games; retained resume checkpoints"), 5)
        self.assertIn("uploaded=true", workflow)
        self.assertIn("uploaded=false", workflow)

    def test_self_play_records_actual_completed_search_metrics(self):
        worker = (ROOT / "src/workers/self_play_shared.ts").read_text()
        for required in ("record.qd.push(analysis.depth)", "record.qn.push(analysis.nodes)",
                         "record.qt.push(Math.round(analysis.elapsedMs))",
                         "saveCheckpoint", "restoreCheckpoint"):
            self.assertIn(required, worker)

if __name__ == "__main__":
    unittest.main()
