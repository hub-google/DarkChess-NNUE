import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
WORKERS_DIR = PROJECT_ROOT / "src" / "workers"
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(WORKERS_DIR))

from board import DarkChessBoardPy, INITIAL_COUNTS, decode_move
from nnue_eval import ModelEvaluator
from search import ChanceSearch, ExactChanceSearch
from tablebase import estimate_raw_states
from train import extract_features, load_model_file
import self_play


class FullModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, board):
        tensor = torch.from_numpy(
            extract_features(board, self.model.input_size)
        ).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(tensor).item())

    def evaluate_many(self, boards):
        features = np.stack(
            [extract_features(board, self.model.input_size) for board in boards]
        )
        with torch.no_grad():
            return (
                self.model(torch.from_numpy(features))
                .squeeze(1)
                .cpu()
                .numpy()
            )


def make_bag(seed):
    bag = np.repeat(np.arange(14, dtype=np.int32), INITIAL_COUNTS)
    np.random.default_rng(seed).shuffle(bag)
    return bag


def reveal_to_hidden(board, target_hidden):
    while int(board.hidden_bitboard).bit_count() > target_hidden:
        moves = [int(move) for move in board.generate_legal_moves()]
        flips = []
        for move in moves:
            from_sq, _, is_flip = decode_move(move)
            if is_flip:
                flips.append((from_sq, move))
        if not flips:
            raise RuntimeError("no flip available while hidden pieces remain")
        _, move = min(flips)
        board.make_move(move, validate=True)
    return board


def make_positions(config):
    positions = {}
    for index, spec in enumerate(config["positions"]):
        board = DarkChessBoardPy(bag=make_bag(config["seed"] + index))
        reveal_to_hidden(board, int(spec["hidden"]))
        positions[spec["name"]] = board
    return positions


def build_transition_samples(seed, count=32):
    board = DarkChessBoardPy(bag=make_bag(seed))
    transitions = []
    for _ in range(count):
        moves = [int(move) for move in board.generate_legal_moves()]
        if not moves:
            break
        # Prefer reveals so the benchmark exercises hidden/global feature deltas.
        flips = [move for move in moves if decode_move(move)[2]]
        move = min(flips) if flips else min(moves)
        from_sq, _, is_flip = decode_move(move)
        flip_piece = int(board.hidden_pieces[from_sq]) if is_flip else None
        parent = board
        child = board.clone()
        child.make_move(move, flip_piece=flip_piece, validate=False)
        transitions.append((parent, child, move, flip_piece))
        board = child
    return transitions


def benchmark_evaluator(model, transitions):
    full = FullModelEvaluator(model)
    incremental = ModelEvaluator(model)

    for parent, _, _, _ in transitions:
        incremental.accumulator(parent)

    repeats = 20
    start = time.perf_counter()
    full_checksum = 0.0
    for _ in range(repeats):
        for _, child, _, _ in transitions:
            full_checksum += full(child)
    full_seconds = time.perf_counter() - start

    start = time.perf_counter()
    incremental_checksum = 0.0
    for _ in range(repeats):
        for parent, child, move, flip_piece in transitions:
            incremental.prepare_child(
                parent,
                child,
                move,
                flip_piece=flip_piece,
            )
            incremental_checksum += incremental(child)
    incremental_seconds = time.perf_counter() - start

    evaluations = repeats * len(transitions)
    return {
        "evaluations": evaluations,
        "full_eval_per_sec": evaluations / full_seconds,
        "incremental_eval_per_sec": evaluations / incremental_seconds,
        "speedup": full_seconds / incremental_seconds,
        "checksum_abs_diff": abs(full_checksum - incremental_checksum),
    }


def benchmark_search(model, positions, config):
    results = []
    for spec in config["positions"]:
        board = positions[spec["name"]]
        row = {"name": spec["name"], "hidden": spec["hidden"]}
        for label, evaluator in (
            ("full", FullModelEvaluator(model)),
            ("incremental", ModelEvaluator(model)),
        ):
            started = time.perf_counter()
            result = ChanceSearch(
                evaluator=evaluator,
                max_depth=int(spec["max_depth"]),
                node_budget=int(spec["node_budget"]),
            ).analyze(board)
            seconds = time.perf_counter() - started
            row[label] = {
                "move": int(result.move),
                "value": float(result.value),
                "nodes": int(result.nodes),
                "seconds": seconds,
                "nodes_per_sec": result.nodes / seconds if seconds else 0.0,
                "completed_depth": int(result.depth),
            }
        row["same_move"] = row["full"]["move"] == row["incremental"]["move"]
        row["value_abs_diff"] = abs(
            row["full"]["value"] - row["incremental"]["value"]
        )
        results.append(row)
    return results


def benchmark_star1(model, seed):
    board = DarkChessBoardPy(bag=make_bag(seed))
    reveal_to_hidden(board, 2)
    evaluator = ModelEvaluator(model)

    rows = {}
    for label, cls in (("star1", ChanceSearch), ("exact", ExactChanceSearch)):
        started = time.perf_counter()
        result = cls(
            evaluator=evaluator,
            max_depth=2,
        ).analyze(board)
        seconds = time.perf_counter() - started
        rows[label] = {
            "move": int(result.move),
            "value": float(result.value),
            "nodes": int(result.nodes),
            "seconds": seconds,
        }
    rows["same_move"] = rows["star1"]["move"] == rows["exact"]["move"]
    rows["value_abs_diff"] = abs(
        rows["star1"]["value"] - rows["exact"]["value"]
    )
    return rows


def summarize_search(rows, label):
    total_nodes = sum(row[label]["nodes"] for row in rows)
    total_seconds = sum(row[label]["seconds"] for row in rows)
    return {
        "positions": len(rows),
        "nodes": int(total_nodes),
        "seconds": float(total_seconds),
        "nodes_per_sec": (
            float(total_nodes) / total_seconds if total_seconds else 0.0
        ),
        "average_completed_depth": (
            sum(row[label]["completed_depth"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "average_move_seconds": (
            total_seconds / len(rows) if rows else 0.0
        ),
    }


def _run_complete_game(evaluator, label, config, budgets, seed):
    np.random.seed(seed)
    record, metrics = self_play.play_game(
        evaluator,
        label,
        np.random.default_rng(seed),
        temperature=0.0,
        explore_plies=0,
        collect_metrics=True,
    )
    metrics["plies"] = int(record["ply"])
    metrics["result"] = float(record["res"])
    return metrics


def benchmark_complete_game(model, config):
    spec = config.get("throughput_game", {})
    env_keys = (
        "SEARCH_NODE_BUDGET_EARLY",
        "SEARCH_NODE_BUDGET_MID",
        "SEARCH_NODE_BUDGET_LATE",
        "OPENING_SEARCH_DEPTH",
    )
    old_env = {key: os.environ.get(key) for key in env_keys}
    budgets = spec.get(
        "node_budgets",
        {"early": 500, "mid": 1000, "late": 2000},
    )
    os.environ["SEARCH_NODE_BUDGET_EARLY"] = str(budgets["early"])
    os.environ["SEARCH_NODE_BUDGET_MID"] = str(budgets["mid"])
    os.environ["SEARCH_NODE_BUDGET_LATE"] = str(budgets["late"])
    os.environ["OPENING_SEARCH_DEPTH"] = str(spec.get("opening_depth", 1))
    try:
        seed = int(config["seed"]) + 500
        full = _run_complete_game(
            FullModelEvaluator(model),
            "benchmark-full",
            config,
            budgets,
            seed,
        )
        hybrid = _run_complete_game(
            ModelEvaluator(model),
            "benchmark-hybrid",
            config,
            budgets,
            seed,
        )
        result = {
            "full": full,
            "hybrid": hybrid,
            "same_plies": full["plies"] == hybrid["plies"],
            "same_result": full["result"] == hybrid["result"],
            "games_per_hour_ratio": (
                hybrid["games_per_hour"] / full["games_per_hour"]
                if full["games_per_hour"]
                else 0.0
            ),
            "average_move_seconds_ratio": (
                hybrid["average_move_seconds"] / full["average_move_seconds"]
                if full["average_move_seconds"]
                else 0.0
            ),
            "node_budgets": {
                "early": int(budgets["early"]),
                "mid": int(budgets["mid"]),
                "late": int(budgets["late"]),
            },
            "note": (
                "Complete-game throughput compares full vs hybrid on the same "
                "seed and reduced benchmark budgets. Production-budget speed "
                "is measured separately on fixed early/mid/late positions."
            ),
        }
        return result
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def main():
    torch.set_num_threads(1)
    config = json.loads(
        (PROJECT_ROOT / "benchmarks" / "search_positions.json").read_text(
            encoding="utf-8"
        )
    )
    model_path = PROJECT_ROOT / "models" / "champion.nnue"
    model = load_model_file(model_path)

    positions = make_positions(config)
    transitions = build_transition_samples(config["seed"] + 100)
    search_rows = benchmark_search(model, positions, config)
    output = {
        "model_input_size": int(model.input_size),
        "evaluator": benchmark_evaluator(model, transitions),
        "search": search_rows,
        "search_summary": {
            "full": summarize_search(search_rows, "full"),
            "incremental": summarize_search(search_rows, "incremental"),
        },
        "complete_game_throughput": benchmark_complete_game(model, config),
        "star1_vs_exact": benchmark_star1(model, config["seed"] + 200),
        "tablebase_raw_state_lower_bounds": {
            str(pieces): estimate_raw_states(pieces)
            for pieces in range(2, 7)
        },
    }

    output_path = PROJECT_ROOT / "benchmark_results.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
