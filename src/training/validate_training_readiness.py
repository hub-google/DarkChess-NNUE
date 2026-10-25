"""Stop production training before a tiny or corrupted replay can replace a champion."""
import argparse
import json
from pathlib import Path


def check_readiness(manifest, min_games=1000):
    games = int(manifest["valid_games"])
    samples = int(manifest["samples"])
    if games < min_games:
        raise RuntimeError(
            f"Only {games} valid games, require >= {min_games}; "
            "refusing to train/promote from an undersized replay window."
        )
    if samples < games:
        raise RuntimeError(f"Too few samples: {samples} for {games} games")
    print(f"[Readiness] READY games={games} samples={samples}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-games", type=int, default=1000)
    args = parser.parse_args()
    check_readiness(
        json.loads(Path(args.manifest).read_text(encoding="utf-8")),
        args.min_games,
    )
