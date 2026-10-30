"""A failed promotion must not reset continuous research training."""
import tempfile
import unittest
from pathlib import Path
import sys

import torch

TRAINING = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING))

from train import CURRENT_INPUT_SIZE, DarkChessNNUE, initialize_challenger, select_training_base
from research_checkpoint import validate_research_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def save_model(path, value):
    model = DarkChessNNUE()
    with torch.no_grad():
        model.fc3.bias.fill_(value)
    torch.save({"format_version": 2, "input_size": CURRENT_INPUT_SIZE,
                "state_dict": model.state_dict()}, path)


class ContinuousResearchTests(unittest.TestCase):
    def test_first_cycle_bootstraps_champion(self):
        with tempfile.TemporaryDirectory() as temp:
            champion = Path(temp) / "champion.nnue"
            research = Path(temp) / "research.nnue"
            save_model(champion, 0.1)
            chosen = select_training_base(str(research), str(champion))
            self.assertEqual(chosen, str(champion))
            self.assertAlmostEqual(initialize_challenger(chosen).fc3.bias.item(), 0.1, places=5)

    def test_later_cycle_inherits_research_even_when_champion_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            champion = Path(temp) / "champion.nnue"
            research = Path(temp) / "research.nnue"
            save_model(champion, 0.1)
            save_model(research, 0.7)
            chosen = select_training_base(str(research), str(champion))
            self.assertEqual(chosen, str(research))
            self.assertAlmostEqual(initialize_challenger(chosen).fc3.bias.item(), 0.7, places=5)

    def test_rejects_invalid_research_instead_of_silently_resetting(self):
        with tempfile.TemporaryDirectory() as temp:
            research = Path(temp) / "research.nnue"
            champion = Path(temp) / "champion.nnue"
            save_model(champion, 0.2)
            save_model(research, 0.3)
            checkpoint = torch.load(research, map_location="cpu", weights_only=True)
            checkpoint["state_dict"]["fc3.bias"].fill_(float("nan"))
            torch.save(checkpoint, research)
            self.assertEqual(select_training_base(str(research), str(champion)), str(research))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                validate_research_checkpoint(research)

    def test_research_checkpoint_shape_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            valid = Path(temp) / "research.nnue"
            save_model(valid, 0.4)
            self.assertEqual(validate_research_checkpoint(valid), valid)
            with self.assertRaises(FileNotFoundError):
                validate_research_checkpoint(Path(temp) / "missing.nnue")

    def test_workflow_commits_model_and_marker_together_before_sprt(self):
        workflow = (ROOT / ".github/workflows/train.yml").read_text(encoding="utf-8")
        self.assertIn("git add models/research.nnue .github/training_last_snapshot.json", workflow)
        self.assertIn("python -u src/training/research_checkpoint.py", workflow)
        production = workflow.split("  train-and-promote:", 1)[1]
        self.assertLess(production.index("cp models/challenger.nnue models/research.nnue"),
                        production.index("node .shared-build/src/workers/sprt_shared.js"))
        self.assertIn("if: steps.validate.outputs.passed == 'true'", workflow)
        self.assertIn("id: production_probe", workflow)
        self.assertIn("--profile production", workflow)
        self.assertIn("--probe-only", workflow)
        self.assertIn("steps.production_probe.outcome == 'success'", workflow)
        self.assertIn("production_full_depth_moves", workflow)
        self.assertNotIn("cp training_snapshot_candidate.json .github/training_last_snapshot.json\n          git config", workflow.split("- name: 保存模型驗證與訓練紀錄")[1])


if __name__ == "__main__":
    unittest.main()
