"""Play paired-deal games against independent MistyBanqi native Rust engine.

This is NOT the same as comparing against the latest Computer Olympiad winner.
Enforces OUR python board as referee with fixed 60-ply and chase rules, and
rejects any mismatching UCI legal move rather than silently accepting it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from board import DarkChessBoardPy, PIECE_COLOR, RED
from misty_uci import choose_misty_move
from search import ChanceSearch, material_evaluate
from sprt_validation import make_bag, score_for_color
from train import load_model_file
from nnue_eval import ModelEvaluator


def play_external_game(model, executable, bag, first_square, model_moves_first,
                       node_budget=500, max_seconds=0.15):
    board = DarkChessBoardPy(bag=bag)
    first_color = int(PIECE_COLOR[int(board.hidden_pieces[first_square])])
    board.make_move((first_square << 5) | first_square, validate=True)
    candidate_color = first_color if model_moves_first else 1 - first_color
    for ply in range(512):
        ended, result = board.is_game_over()
        if ended:
            return score_for_color(result, candidate_color), ply + 1
        if int(board.side_to_move) == candidate_color:
            search = ChanceSearch(
                evaluator=model, max_depth=8,
                node_budget=node_budget, max_seconds=max_seconds,
            ).analyze(board)
            move = search.move
        else:
            move = choose_misty_move(board, executable, node_budget, max_seconds)
        board.make_move(int(move), validate=True)
    raise RuntimeError("external paired game exceeded 512 plies without a terminal state")


def benchmark(model, executable, pairs, node_budget, max_seconds, seed):
    rng = np.random.default_rng(seed)
    total = 0.0
    details = []
    for index in range(pairs):
        bag = make_bag(rng)
        opening = int(rng.integers(0, 32))
        first, first_plies = play_external_game(
            model, executable, bag, opening, True, node_budget, max_seconds,
        )
        second, second_plies = play_external_game(
            model, executable, bag, opening, False, node_budget, max_seconds,
        )
        points = first + second
        total += points
        details.append({
            "pair": index + 1, "candidate_points": points,
            "first_game_plies": first_plies, "second_game_plies": second_plies,
        })
        print(f"[External Arena] pair={index+1}/{pairs} model_points={points}/2", flush=True)
    return {
        "candidate": "NNUE champion", "opponent": "MistyBanqi 0.2.7 pinned SHA",
        "pairs": pairs, "games": 2 * pairs,
        "candidate_points": total,
        "candidate_score_rate": total / (2 * pairs),
        "node_budget_each": node_budget, "seconds_each": max_seconds,
        "seed": seed, "details": details,
        "warning": "Small trial, distinct repetition semantics, no world ranking claim.",
    }


class HybridEvaluator:
    """Experimentally blend existing champion with independent material logic."""
    def __init__(self, value_evaluator, material_weight=0.25):
        self.value_evaluator = value_evaluator
        self.material_weight = float(material_weight)

    def __call__(self, board):
        return (
            (1.0 - self.material_weight) * self.value_evaluator(board)
            + self.material_weight * material_evaluate(board)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misty", required=True)
    ap.add_argument("--candidate-mode", choices=["champion", "material", "hybrid"], default="champion")
    ap.add_argument("--champion", default="models/champion.nnue")
    ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--node-budget", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--output", default="external_arena_results.json")
    args = ap.parse_args()
    if args.pairs < 1 or args.node_budget < 1 or args.max_seconds <= 0:
        ap.error("invalid paired experiment settings")
    torch.set_num_threads(1)
    champion = ModelEvaluator(load_model_file(args.champion))
    model = (
        material_evaluate if args.candidate_mode == "material" else
        HybridEvaluator(champion) if args.candidate_mode == "hybrid" else
        champion
    )
    started = time.monotonic()
    report = benchmark(model, args.misty, args.pairs, args.node_budget, args.max_seconds, args.seed)
    report["wall_seconds"] = round(time.monotonic() - started, 1)
    report["candidate_mode"] = args.candidate_mode
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[External Arena] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
