import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import gzip
import json
import zlib
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from board import DarkChessBoardPy, INITIAL_COUNTS

CURRENT_INPUT_SIZE = 498
LEGACY_INPUT_SIZE = 494
MODEL_VERSION = 2

# --- NNUE Model Definition ---
class DarkChessNNUE(nn.Module):
    def __init__(self, input_size=CURRENT_INPUT_SIZE):
        super(DarkChessNNUE, self).__init__()
        self.input_size = input_size
        # 32 squares * 15 piece types (14 visible + 1 hidden) = 480
        # + 14 bag counts + side-to-move(2) + draw state(2) = 498.
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.clamp(torch.relu(self.fc1(x)), max=1.0)
        x = torch.clamp(torch.relu(self.fc2(x)), max=1.0)
        return torch.tanh(self.fc3(x))


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_model_file(path):
    """Load either the legacy 494-input model or the current model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_state_dict(checkpoint)
    input_size = int(state_dict["fc1.weight"].shape[1])
    if input_size not in (LEGACY_INPUT_SIZE, CURRENT_INPUT_SIZE):
        raise ValueError(f"unsupported model input size: {input_size}")
    model = DarkChessNNUE(input_size=input_size)
    model.load_state_dict(state_dict)
    return model


def initialize_challenger(champion_path):
    """Continue from champion, upgrading the legacy input layer when needed."""
    model = DarkChessNNUE()
    if not os.path.exists(champion_path):
        print("No champion model found; starting a new v2 model.")
        return model

    champion = load_model_file(champion_path)
    if champion.input_size == CURRENT_INPUT_SIZE:
        model.load_state_dict(champion.state_dict())
        print(f"Loaded champion weights from {champion_path}.")
        return model

    # Preserve all learned legacy weights and initialize new public-state
    # features to zero influence. Training can then learn them safely.
    with torch.no_grad():
        model.fc1.weight[:, :LEGACY_INPUT_SIZE].copy_(champion.fc1.weight)
        bag_scale = torch.tensor(INITIAL_COUNTS, dtype=model.fc1.weight.dtype)
        model.fc1.weight[:, 480:494].mul_(bag_scale)
        model.fc1.weight[:, LEGACY_INPUT_SIZE:].zero_()
        model.fc1.bias.copy_(champion.fc1.bias)
        model.fc2.load_state_dict(champion.fc2.state_dict())
        model.fc3.load_state_dict(champion.fc3.state_dict())
    print(f"Upgraded legacy champion from {LEGACY_INPUT_SIZE} to {CURRENT_INPUT_SIZE} inputs.")
    return model

# --- TD-Learning Loss ---
def td_loss(predictions, absolute_results):
    """
    Target = 最終勝負 (For now, simplified without MCTS root eval)
    """
    return nn.MSELoss()(predictions, absolute_results)

# --- Feature Extraction ---
def extract_features(board, input_size=CURRENT_INPUT_SIZE):
    if input_size not in (LEGACY_INPUT_SIZE, CURRENT_INPUT_SIZE):
        raise ValueError(f"unsupported feature size: {input_size}")
    features = np.zeros(input_size, dtype=np.float32)
    for p in range(14):
        bb = int(board.piece_bitboards[p])
        for sq in range(32):
            if (bb >> sq) & 1:
                features[sq * 15 + p] = 1.0
                
    hb = int(board.hidden_bitboard)
    for sq in range(32):
        if (hb >> sq) & 1:
            features[sq * 15 + 14] = 1.0

    # Use public counts only. Dividing by the initial inventory keeps all
    # inputs in [0, 1] while preserving the exact remaining count.
    if input_size == LEGACY_INPUT_SIZE:
        features[480:494] = board.remaining_counts.astype(np.float32)
    else:
        features[480:494] = (
            board.remaining_counts.astype(np.float32)
            / INITIAL_COUNTS.astype(np.float32)
        )

    if input_size >= CURRENT_INPUT_SIZE:
        if board.side_to_move == 0:
            features[494] = 1.0
        elif board.side_to_move == 1:
            features[495] = 1.0
        features[496] = min(float(board.half_move_clock) / 60.0, 1.0)
        features[497] = min(float(board.repetition_count()) / 3.0, 1.0)
    return features

class DarkChessDataset(IterableDataset):
    def __init__(
        self,
        files,
        input_size=CURRENT_INPUT_SIZE,
        max_positions_per_game=4,
        max_samples=2_000_000,
    ):
        self.files = files
        self.input_size = input_size
        self.max_positions_per_game = max_positions_per_game
        self.max_samples = max_samples
        
    def __iter__(self):
        files = list(self.files)
        np.random.shuffle(files)
        yielded = 0
        for f in files:
            try:
                with gzip.open(f, 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        if not line.strip(): continue
                        game = json.loads(line)
                        version = str(game.get("ver", ""))
                        if not version.startswith("v2."):
                            # v1 games were generated by the always-capture
                            # random heuristic and would dominate the improved
                            # policy with systematically biased targets.
                            continue
                        board = DarkChessBoardPy(bag=game['hid'])
                        res = float(game['res'])
                        if res not in (-1.0, 0.0, 1.0):
                            raise ValueError(f"invalid result: {res}")
                        moves = game.get('mov')
                        if not isinstance(moves, list) or not moves:
                            raise ValueError("game has no moves")
                        root_values = game.get('q')
                        if root_values is not None and len(root_values) != len(moves):
                            raise ValueError("q and mov lengths differ")

                        # Validate the complete record before yielding any
                        # samples so a late illegal move cannot partially
                        # poison an optimizer step.
                        validation_board = DarkChessBoardPy(bag=game['hid'])
                        for move in moves:
                            validation_board.make_move(int(move), validate=True)
                        over, replay_result = validation_board.is_game_over()
                        if not over or replay_result != res:
                            raise ValueError(
                                f"recorded result {res} does not match replay "
                                f"terminal state {(over, replay_result)}"
                            )

                        eligible = np.arange(1, len(moves), dtype=np.int32)
                        if len(eligible) > self.max_positions_per_game:
                            seed = zlib.crc32(str(game["id"]).encode("utf-8"))
                            game_rng = np.random.default_rng(seed)
                            eligible = game_rng.choice(
                                eligible,
                                size=self.max_positions_per_game,
                                replace=False,
                            )
                        selected_plies = set(int(ply) for ply in eligible)

                        board = DarkChessBoardPy(bag=game['hid'])
                        for ply, move in enumerate(moves):
                            side = board.side_to_move
                            if side != 2 and ply in selected_plies:
                                target = res
                                if root_values is not None:
                                    root_value = float(root_values[ply])
                                    if not -1.0 <= root_value <= 1.0:
                                        raise ValueError(f"invalid root value: {root_value}")
                                    target = 0.5 * res + 0.5 * root_value
                                feat = extract_features(board, self.input_size)
                                yield torch.tensor(feat), torch.tensor([target], dtype=torch.float32)
                                yielded += 1
                                if yielded >= self.max_samples:
                                    return
                            board.make_move(int(move), validate=True)
            except Exception as e:
                print(f"Error reading {f}: {e}")

def main():
    print("Initializing DarkChess NNUE Training Pipeline...")
    models_dir = os.environ.get("MODELS_DIR", "models")
    champion_path = os.environ.get(
        "CHAMPION_PATH",
        os.path.join(models_dir, "champion.nnue"),
    )
    model = initialize_challenger(champion_path)
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.0001"))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    
    # 1. Read files downloaded from Hugging Face into the local datasets/ directory
    dataset_dir = os.environ.get("DATASET_DIR", "datasets")
    files = glob.glob(os.path.join(dataset_dir, "**/*.jsonl.gz"), recursive=True)
    print(f"Discovered {len(files)} batches locally in {dataset_dir}.")
    if not files:
        raise RuntimeError("No training data found; refusing to create a challenger.")
    
    # Replay Buffer mechanism (Sliding window up to 500k games)
    print("Replay buffer configured. Max capacity: 500,000 games.")
    
    # Four positions per game guarantees that the 500k-game replay window can
    # be traversed completely within the default two-million-sample budget.
    max_positions = int(os.environ.get("MAX_POSITIONS_PER_GAME", "4"))
    max_samples = int(os.environ.get("MAX_TRAINING_SAMPLES", "2000000"))
    if max_positions * 500_000 > max_samples:
        raise RuntimeError(
            "Training sample budget cannot cover the complete 500,000-game "
            "replay window; lower MAX_POSITIONS_PER_GAME or raise "
            "MAX_TRAINING_SAMPLES."
        )
    dataset = DarkChessDataset(
        files,
        input_size=CURRENT_INPUT_SIZE,
        max_positions_per_game=max_positions,
        max_samples=max_samples,
    )
    batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
    epochs = int(os.environ.get("TRAINING_EPOCHS", "3"))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print(
        f"Hyperparameters: lr={learning_rate}, batch_size={batch_size}, "
        f"epochs={epochs}"
    )
    print("Training loop ready.")

    # REAL TRAINING LOOP
    print("Starting training epochs...")
    model.train()
    
    total_loss = 0
    batches = 0
    for epoch in range(epochs):
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = td_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            if batches % 100 == 0:
                print(f"Epoch {epoch} | Batch {batches} | Loss: {total_loss/100:.4f}")
                total_loss = 0

    if batches == 0:
        raise RuntimeError("Training produced zero valid batches; refusing to save a challenger.")
                
    print("Training complete.")

    # 2. Save the newly trained model to models/challenger.nnue
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.environ.get(
        "CHALLENGER_PATH",
        os.path.join(models_dir, "challenger.nnue"),
    )
    
    torch.save(
        {
            "format_version": MODEL_VERSION,
            "input_size": CURRENT_INPUT_SIZE,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"Model saved successfully to {model_path}")

if __name__ == '__main__':
    main()
