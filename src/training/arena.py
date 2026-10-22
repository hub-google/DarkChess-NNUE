"""Deterministic CPU paired-arena for external baselines and archived champions.

This is an evaluation tool. It NEVER trains or auto-promotes a model.
Scores are reported in paired two-game blocks using the identical hidden bag
and initial reveal; 2-10 pairs are smoke tests, not evidence of superiority.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from nnue_eval import ModelEvaluator
from search import material_evaluate
from sprt_validation import make_bag, play_game
from train import load_model_file


def load_evaluator(path):
    if path == "material":
        return material_evaluate
    if not Path(path).is_file():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    return ModelEvaluator(load_model_file(path))


def elo_from_score(score):
    """Finite descriptive Elo difference; do not interpret as a rating."""
    p = max(1e-5, min(1 - 1e-5, float(score)))
    return 400 * math.log10(p / (1 - p))


def paired_benchmark(candidate, baseline, pairs=2, seed=20261022,
                     depth=4, node_budget=500, max_seconds=0.2):
    if pairs < 1 or max_seconds <= 0 or node_budget < 1:
        raise ValueError("invalid paired benchmark settings")
    rng = np.random.default_rng(seed)
    pair_scores = []
    for pair in range(pairs):
        bag = make_bag(rng)
        square = int(rng.integers(0, 32))
        first = play_game(
            candidate, baseline, bag, square, depth, node_budget, max_seconds
        )
        reverse = play_game(
            baseline, candidate, bag, square, depth, node_budget, max_seconds
        )
        score = first + (1 - reverse)
        pair_scores.append(float(score))
        print(f"[Arena] Pair {pair + 1}/{pairs}: candidate points={score:.1f}/2")
    average = sum(pair_scores) / (2 * len(pair_scores))
    return {
        "paired_games": 2 * pairs,
        "pairs": pairs,
        "candidate_points": sum(pair_scores),
        "candidate_score_rate": average,
        "descriptive_elo_difference": elo_from_score(average),
        "pair_scores": pair_scores,
        "note": "Paired score only; small samples and no external rating pool cannot establish world ranking.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="models/champion.nnue")
    parser.add_argument("--baseline", default="material")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--node-budget", type=int, default=500)
    parser.add_argument("--max-seconds", type=float, default=0.2)
    parser.add_argument("--output", default="arena_results.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    candidate, baseline = load_evaluator(args.candidate), load_evaluator(args.baseline)
    result = paired_benchmark(candidate, baseline, args.pairs, args.seed,
                              args.depth, args.node_budget, args.max_seconds)
    result["candidate"] = args.candidate
    result["baseline"] = args.baseline
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[Arena] " + json.dumps(result))


if __name__ == "__main__":
    main()
