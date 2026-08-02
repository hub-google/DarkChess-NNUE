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
from search import (  # noqa: E402
    ChanceSearch,
    material_evaluate,
    select_first_flip,
    select_move,
)
from train import extract_features, load_model_file  # noqa: E402


class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, board):
        features = extract_features(board, self.model.input_size)
        tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(tensor).item())

    def evaluate_many(self, boards):
        features = np.stack(
            [extract_features(board, self.model.input_size) for board in boards]
        )
        tensor = torch.from_numpy(features)
        with torch.no_grad():
            return self.model(tensor).squeeze(1).cpu().numpy()


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

    model = load_model_file(champion_path)
    print(
        f"[Self-Play] Loaded champion with {model.input_size} input features "
        f"from {champion_path}."
    )
    return ModelEvaluator(model), f"champion-{model.input_size}"


def play_game(evaluator, model_version, rng, search_depth, temperature, explore_plies):
    board = DarkChessBoardPy()
    record = {
        "id": str(uuid.uuid4()),
        "ts": int(time.time() * 1000),
        "ver": "v2.0.0-belief-search",
        "model": model_version,
        "hid": [int(piece) for piece in board.hidden_pieces],
        "mov": [],
        "q": [],
        "res": 0.0,
        "ply": 0,
    }

    opening = ChanceSearch(evaluator=evaluator, max_depth=search_depth).analyze_first_flip(board)
    first_move = select_first_flip(opening, temperature=temperature, rng=rng)
    record["mov"].append(first_move)
    record["q"].append(float(opening.move_values[first_move]))
    board.make_move(first_move, validate=False)

    while record["ply"] < 512:
        over, result = board.is_game_over()
        if over:
            record["res"] = float(result)
            break

        search = ChanceSearch(evaluator=evaluator, max_depth=search_depth)
        analysis = search.analyze(board)
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
        board.make_move(chosen, validate=False)
        record["ply"] += 1
    else:
        # This guard should be unreachable when repetition and 60-ply rules
        # work, but it prevents a bad game from blocking a worker forever.
        record["res"] = 0.0

    record["ply"] = len(record["mov"])
    return record


def run_batch(batch_size, output_dir, evaluator, model_version, rng):
    search_depth = int(os.environ.get("MAX_SEARCH_DEPTH", "1"))
    temperature = float(os.environ.get("SELF_PLAY_TEMPERATURE", "0.8"))
    explore_plies = int(os.environ.get("EXPLORE_PLIES", "20"))

    games = []
    for index in range(batch_size):
        games.append(
            play_game(
                evaluator,
                model_version,
                rng,
                search_depth,
                temperature,
                explore_plies,
            )
        )
        if (index + 1) % 10 == 0:
            print(f"[Self-Play] Generated {index + 1}/{batch_size} games.")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(uuid.uuid4())
    output_path = output_dir / f"data_{int(time.time() * 1000)}_{batch_id}.jsonl.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for game in games:
            handle.write(json.dumps(game, separators=(",", ":")) + "\n")
    print(f"[Self-Play] Wrote {len(games)} games to {output_path}")


def main():
    batch_size = int(os.environ.get("BATCH_SIZE", "50"))
    num_batches = int(os.environ.get("NUM_BATCHES", "1"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "output_data"))
    seed = int(os.environ.get("SELF_PLAY_SEED", str(time.time_ns() % (2**32))))
    rng = np.random.default_rng(seed)
    evaluator, model_version = load_evaluator()

    print(
        f"[Self-Play] Starting {num_batches} batches x {batch_size} games "
        f"with seed {seed}."
    )
    for _ in range(num_batches):
        run_batch(batch_size, output_dir, evaluator, model_version, rng)


if __name__ == "__main__":
    main()
