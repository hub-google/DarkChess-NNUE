import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from policy_train import PolicyNet, policy_loss, train, transform_actions, transform_legal_mask
from train import transform_square, CURRENT_INPUT_SIZE


class PolicyTeacherTests(unittest.TestCase):
    def test_move_symmetry_round_trip(self):
        original = np.array([1 * 32 + 9, 8 * 32 + 8, -1])
        twice = transform_actions(transform_actions(original, 1), 1)
        np.testing.assert_array_equal(twice, original)
        self.assertEqual(int(transform_actions(original, 1)[1]),
                         transform_square(8, 1) * 33)

    def test_soft_cross_entropy_is_finite(self):
        logits = torch.zeros((1, 1024))
        loss = policy_loss(logits, torch.tensor([[10, 20, -1]]),
                           torch.tensor([[0.25, 0.75, 0.0]]))
        self.assertTrue(torch.isfinite(loss).item())

    def test_illegal_action_logit_does_not_change_loss(self):
        actions = torch.tensor([[33, 34, -1]])
        probs = torch.tensor([[0.25, 0.75, 0.0]])
        legal = torch.zeros((1, 1024), dtype=torch.bool)
        legal[0, [33, 34, 35]] = True
        logits = torch.zeros((1, 1024))
        base = policy_loss(logits, actions, probs, legal)
        logits[0, 800] = 1000
        self.assertAlmostEqual(float(base), float(policy_loss(logits, actions, probs, legal)), places=5)

    def test_policy_rejects_illegal_teacher(self):
        logits = torch.zeros((1, 1024))
        mask = torch.zeros((1, 1024), dtype=torch.bool)
        mask[0, 33] = True
        with self.assertRaisesRegex(ValueError, "illegal"):
            policy_loss(logits, torch.tensor([[99]]), torch.tensor([[1.]]), mask)

    def test_legal_mask_symmetry(self):
        packed = np.zeros(128, dtype=np.uint8)
        packed[33 // 8] |= (1 << (33 % 8))
        for sym in range(4):
            mask = transform_legal_mask(packed, sym)
            self.assertTrue(mask[int(transform_actions(np.array([33]), sym)[0])])
            self.assertEqual(int(mask.sum()), 1)

    def test_cpu_trains_policy_checkpoint_from_cache(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            cache = directory / "cache"
            cache.mkdir()
            feat = np.zeros((2, CURRENT_INPUT_SIZE), np.float16)
            actions = np.full((2, 16), -1, np.int16)
            weights = np.zeros((2, 16), np.float16)
            legal_action_bits = np.zeros((2, 128), np.uint8)
            legal_action_bits[:, 33 // 8] |= (1 << (33 % 8))
            actions[:, 0] = 33
            weights[:, 0] = 1
            np.savez_compressed(cache / "cache_0000.npz",
                                features=feat, policy_actions=actions, policy_probs=weights,
                                legal_action_bits=legal_action_bits)
            output = directory / "challenger.policy"
            trained = train(cache, directory / "missing.policy", output, epochs=1, batch_size=2)
            self.assertEqual(trained, 2)
            checkpoint = torch.load(output, weights_only=True)
            model = PolicyNet()
            model.load_state_dict(checkpoint["state_dict"])
            self.assertEqual(model(torch.zeros(1, CURRENT_INPUT_SIZE)).shape, (1, 1024))
