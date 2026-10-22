import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "workers"))
import self_play
from search import material_evaluate


class DiverseSelfPlayTests(unittest.TestCase):
    def test_bootstrap_baseline_is_selected(self):
        champion = (lambda board: 0.0, "champion")
        rng = np.random.default_rng(1)
        selected = self_play.choose_opponent(
            champion, [], rng, bootstrap_probability=1.0,
        )
        self.assertIs(selected[0], material_evaluate)
        self.assertEqual(selected[1], "bootstrap-material")

    def test_champion_without_baseline_or_archives(self):
        champion = (lambda board: 0.0, "champion")
        rng = np.random.default_rng(1)
        selected = self_play.choose_opponent(
            champion, [], rng, bootstrap_probability=0,
        )
        self.assertIs(selected, champion)

    def test_archived_opponent_can_be_selected(self):
        champion = (lambda board: 0.0, "champion")
        archive = (lambda board: 0.1, "archive")
        rng = np.random.default_rng(1)
        selected = self_play.choose_opponent(
            champion, [archive], rng,
            bootstrap_probability=0, archive_probability=1,
        )
        self.assertIs(selected, archive)
