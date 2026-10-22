"""Optional CPU policy head: distil sparse public-state search root distributions.

Value NNUE remains unchanged/compatible with the browser. The policy head is
a separate experimental checkpoint. It does not control promotion or inference
unless explicitly enabled by a future measured comparison.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train import CURRENT_INPUT_SIZE, augment_features, transform_square


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CURRENT_INPUT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 32),
        )

    def forward(self, features):
        return self.net(features)


def transform_actions(actions, symmetry):
    """Move encoding is (from << 5) | to; flip means from == to."""
    transformed = np.array(actions, dtype=np.int64, copy=True)
    for index, move in enumerate(transformed):
        if move < 0:
            continue
        from_sq, to_sq = int(move) >> 5, int(move) & 31
        transformed[index] = (transform_square(from_sq, symmetry) << 5) | transform_square(to_sq, symmetry)
    return transformed


def policy_loss(logits, actions, probabilities):
    """Soft cross entropy with padding excluded, over the 1024 action IDs."""
    mask = actions >= 0
    safe_actions = actions.clamp(min=0)
    selected_logp = F.log_softmax(logits, dim=1).gather(1, safe_actions)
    mass = probabilities * mask.to(probabilities.dtype)
    mass = mass / mass.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return -(mass * selected_logp).sum(dim=1).mean()


def train(cache_dir, champion, challenger, epochs=1, max_samples=100000, batch_size=128):
    cache_files = sorted(glob.glob(str(Path(cache_dir) / "cache_*.npz")))
    if not cache_files:
        raise RuntimeError("No binary cache shards for policy training")
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261022)
    model = PolicyNet()
    if Path(champion).is_file():
        data = torch.load(champion, weights_only=True, map_location="cpu")
        model.load_state_dict(data["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    trained = 0
    model.train()

    for _ in range(epochs):
        if trained >= max_samples:
            break
        for path in cache_files:
            with np.load(path) as shard:
                if "policy_actions" not in shard or "policy_probs" not in shard:
                    continue
                probs = shard["policy_probs"]
                order = np.flatnonzero(probs.sum(axis=1) > 0)
                rng.shuffle(order)
                for offset in range(0, len(order), batch_size):
                    if trained >= max_samples:
                        break
                    subset = order[offset:offset + min(batch_size, max_samples - trained)]
                    x = []
                    actions = []
                    weights = []
                    for index in subset:
                        feat = shard["features"][index].astype(np.float32)
                        symmetry = int(rng.integers(0, 4))
                        color_swap = bool(rng.integers(0, 2))
                        transformed, _ = augment_features(
                            feat, 0.0, transform=symmetry, color_swap=color_swap
                        )
                        x.append(transformed)
                        actions.append(transform_actions(shard["policy_actions"][index], symmetry))
                        weights.append(shard["policy_probs"][index].astype(np.float32))
                    if not x:
                        continue
                    optimizer.zero_grad()
                    logits = model(torch.from_numpy(np.stack(x)))
                    loss = policy_loss(
                        logits,
                        torch.from_numpy(np.stack(actions)),
                        torch.from_numpy(np.stack(weights)),
                    )
                    loss.backward()
                    optimizer.step()
                    trained += len(x)

    if trained == 0:
        print("[Policy] No valid search-teacher positions; policy training skipped.")
        return 0
    Path(challenger).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format_version": 1, "input_size": CURRENT_INPUT_SIZE,
         "action_size": 1024, "state_dict": model.state_dict(), "samples": trained},
        challenger,
    )
    print(f"[Policy] Trained {trained} positions; checkpoint={challenger}")
    return trained


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="training_cache")
    parser.add_argument("--champion", default="models/champion.policy")
    parser.add_argument("--challenger", default="models/challenger.policy")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()
    train(args.cache_dir, args.champion, args.challenger, args.epochs, args.max_samples)
