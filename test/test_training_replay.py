import sys
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from replay_format import is_supported_replay_version


class ReplayVersionTests(unittest.TestCase):
    def test_accepts_current_perpetual_chase_rules(self):
        self.assertTrue(is_supported_replay_version("v2.1.0-perpetual-chase"))

    def test_rejects_old_threefold_draw_rules(self):
        self.assertFalse(is_supported_replay_version("v2.0.0-belief-search"))
        self.assertFalse(is_supported_replay_version("v1.0.0"))


if __name__ == "__main__":
    unittest.main()
