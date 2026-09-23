import gzip
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from board import DarkChessBoardPy  # noqa: E402
from replay_format import CURRENT_REPLAY_VERSION  # noqa: E402
from search import (  # noqa: E402
    ChanceSearch,
    material_evaluate,
    select_first_flip,
    select_move,
)
from nnue_eval import ModelEvaluator  # noqa: E402
from train import load_model_file  # noqa: E402

SLOW_SEARCH_SECONDS = 300



def load_evaluator():
    champion_path = Path(
        os.environ.get(
            "CHAMPION_PATH",
            str(PROJECT_ROOT / "models" / "champion.nnue"),
        )
    )
    if not champion_path.exists():
        print("[Self-Play] No champion found; using public-state material bootstrap.")
        return material_evaluate, "bootstrap-material"

    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    model = load_model_file(champion_path)
    print(
        f"[Self-Play] Loaded champion with {model.input_size} input features "
        f"from {champion_path}."
    )
    return ModelEvaluator(model), f"champion-{model.input_size}"


def choose_search_depth(hidden_count):
    """Use deeper searches only after enough private information is revealed."""
    if hidden_count >= 24:
        return 3
    if hidden_count >= 12:
        return 10
    return 12


def choose_node_budget(hidden_count):
    if hidden_count >= 24:
        return int(os.environ.get("SEARCH_NODE_BUDGET_EARLY", "12000"))
    if hidden_count >= 12:
        return int(os.environ.get("SEARCH_NODE_BUDGET_MID", "50000"))
    return int(os.environ.get("SEARCH_NODE_BUDGET_LATE", "120000"))


def choose_opening_depth():
    """
    The fully hidden opening has little actionable information and only eight
    geometric square classes. Spend the CPU budget later in the game.
    """
    return max(1, int(os.environ.get("OPENING_SEARCH_DEPTH", "1")))


def play_game(evaluator, model_version, rng, temperature, explore_plies):
    game_started = time.perf_counter()
    board = DarkChessBoardPy()
    record = {
        "id": str(uuid.uuid4()),
        "ts": int(time.time() * 1000),
        "ver": CURRENT_REPLAY_VERSION,
        "model": model_version,
        "hid": [int(piece) for piece in board.hidden_pieces],
        "mov": [],
        "q": [],
        "v": [],
        "res": 0.0,
        "ply": 0,
    }

    total_nodes = 0
    total_search_seconds = 0.0
    slowest_search = None

    opening_depth = choose_opening_depth()
    active_depth = opening_depth
    print(
        f"[Self-Play] Game started: id={record['id']} ply=1 hidden=32 "
        f"depth={opening_depth}."
    )
    search_started = time.perf_counter()
    opening = ChanceSearch(
        evaluator=evaluator,
        max_depth=opening_depth,
    ).analyze_first_flip(board)
    opening_seconds = time.perf_counter() - search_started
    total_nodes += opening.nodes
    total_search_seconds += opening_seconds
    slowest_search = (opening_seconds, 1, opening_depth, opening.nodes)
    first_move = select_first_flip(opening, temperature=temperature, rng=rng)
    record["mov"].append(first_move)
    record["q"].append(float(opening.move_values[first_move]))
    record["v"].append(float(np.clip(evaluator(board), -1.0, 1.0)))
    board.make_move(first_move, validate=False)

    while record["ply"] < 512:
        over, result = board.is_game_over()
        if over:
            record["res"] = float(result)
            break

        move_number = len(record["mov"]) + 1
        hidden_count = int(board.hidden_bitboard).bit_count()
        search_depth = choose_search_depth(hidden_count)
        if search_depth != active_depth:
            print(
                f"[Self-Play] Depth transition: game={record['id']} "
                f"ply={move_number} hidden={hidden_count} depth={search_depth}."
            )
            active_depth = search_depth
        node_budget = choose_node_budget(hidden_count)
        static_value = float(np.clip(evaluator(board), -1.0, 1.0))
        search_started = time.perf_counter()
        search = ChanceSearch(
            evaluator=evaluator,
            max_depth=search_depth,
            node_budget=node_budget,
        )
        analysis = search.analyze(board)
        search_seconds = time.perf_counter() - search_started
        total_nodes += analysis.nodes
        total_search_seconds += search_seconds
        if slowest_search is None or search_seconds > slowest_search[0]:
            slowest_search = (
                search_seconds,
                move_number,
                search_depth,
                analysis.nodes,
            )
        if search_seconds >= SLOW_SEARCH_SECONDS:
            print(
                f"[Self-Play] Search exceeded {SLOW_SEARCH_SECONDS}s: "
                f"game={record['id']} "
                f"ply={move_number} hidden={hidden_count} "
                f"depth={analysis.depth}/{search_depth} nodes={analysis.nodes} "
                f"seconds={search_seconds:.1f}."
            )
        current_temperature = (
            temperature if record["ply"] < explore_plies else 0.0
        )
        chosen = select_move(
            analysis,
            board.side_to_move,
            temperature=current_temperature,
            rng=rng,
        )
        record["mov"].append(int(chosen))
        record["q"].append(float(analysis.move_values[chosen]))
        record["v"].append(static_value)
        board.make_move(chosen, validate=False)
        record["ply"] += 1
    else:
        raise RuntimeError(
            "self-play exceeded 512 plies without reaching a terminal state"
        )

    record["ply"] = len(record["mov"])
    elapsed = time.perf_counter() - game_started
    slow_seconds, slow_ply, slow_depth, slow_nodes = slowest_search
    nodes_per_second = total_nodes / total_search_seconds if total_search_seconds else 0.0
    print(
        f"[Self-Play] Game complete: id={record['id']} plies={record['ply']} "
        f"result={record['res']:+.1f} seconds={elapsed:.1f} nodes={total_nodes} "
        f"nodes_per_second={nodes_per_second:.0f} slowest_ply={slow_ply} "
        f"slowest_depth={slow_depth} slowest_nodes={slow_nodes} "
        f"slowest_seconds={slow_seconds:.1f}."
    )
    return record


def run_batch(batch_size, output_dir, evaluator, model_version, rng):
    temperature = float(os.environ.get("SELF_PLAY_TEMPERATURE", "0.8"))
    explore_plies = int(os.environ.get("EXPLORE_PLIES", "20"))

    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(batch_size):
        print(f"[Self-Play] Starting game {index + 1}/{batch_size}.")
        game = play_game(
            evaluator,
            model_version,
            rng,
            temperature,
            explore_plies,
        )
        output_path = output_dir / f"data_{int(time.time() * 1000)}_{game['id']}.jsonl.gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(game, separators=(",", ":")) + "\n")
        print(f"[Self-Play] Saved game {index + 1}/{batch_size} to {output_path}.")


def main():
    batch_size = int(os.environ.get("BATCH_SIZE", "50"))
    num_batches = int(os.environ.get("NUM_BATCHES", "1"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "output_data"))
    seed = int(os.environ.get("SELF_PLAY_SEED", str(time.time_ns() % (2**32))))
    rng = np.random.default_rng(seed)
    evaluator, model_version = load_evaluator()

    print(
        f"[Self-Play] Starting {num_batches} batches x {batch_size} games "
        f"with seed {seed}; opening depth={choose_opening_depth()}, "
        f"adaptive depths after first flip: hidden 24-31=3, 12-23=10, 0-11=12."
    )
    for _ in range(num_batches):
        run_batch(batch_size, output_dir, evaluator, model_version, rng)


if __name__ == "__main__":
    main()
