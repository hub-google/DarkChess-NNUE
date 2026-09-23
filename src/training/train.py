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
from replay_format import is_supported_replay_version

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


BOARD_ROWS = 4
BOARD_COLS = 8
SYM_IDENTITY = 0
SYM_LEFT_RIGHT = 1
SYM_UP_DOWN = 2
SYM_ROTATE_180 = 3
SYMMETRY_COUNT = 4


def transform_square(square, transform):
    """Map a square through a symmetry of the 4x8 rectangle."""
    row, col = divmod(int(square), BOARD_COLS)
    if transform in (SYM_LEFT_RIGHT, SYM_ROTATE_180):
        col = BOARD_COLS - 1 - col
    if transform in (SYM_UP_DOWN, SYM_ROTATE_180):
        row = BOARD_ROWS - 1 - row
    return row * BOARD_COLS + col


def augment_features(features, target, transform=SYM_IDENTITY, color_swap=False):
    """
    Apply an exact game symmetry without duplicating replay files.

    Geometry keeps the Red-perspective target unchanged. Swapping Red/Black
    piece identities also swaps side-to-move and negates the Red-perspective
    value target.
    """
    transformed = np.zeros_like(features)

    for square in range(32):
        mapped = transform_square(square, transform)
        src = square * 15
        dst = mapped * 15
        transformed[dst:dst + 15] = features[src:src + 15]

    transformed[480:494] = features[480:494]
    if len(features) > 494:
        transformed[494:] = features[494:]

    if color_swap:
        for square in range(32):
            base = square * 15
            red_channels = transformed[base:base + 7].copy()
            transformed[base:base + 7] = transformed[base + 7:base + 14]
            transformed[base + 7:base + 14] = red_channels

        red_counts = transformed[480:487].copy()
        transformed[480:487] = transformed[487:494]
        transformed[487:494] = red_counts

        if len(transformed) >= CURRENT_INPUT_SIZE:
            red_to_move = float(transformed[494])
            transformed[494] = transformed[495]
            transformed[495] = red_to_move

        target = -float(target)

    return transformed, float(target)


def _env_enabled(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def select_training_plies(game, moves, max_positions):
    """
    Reserve half the per-game sample budget for positions where search most
    disagrees with the raw NNUE value; use deterministic random coverage for
    the other half. Legacy replays without v keep the old random behavior.
    """
    eligible = np.arange(1, len(moves), dtype=np.int32)
    if len(eligible) <= max_positions:
        return eligible

    seed = zlib.crc32(str(game["id"]).encode("utf-8"))
    game_rng = np.random.default_rng(seed)
    root_values = game.get("q")
    static_values = game.get("v")

    if (
        isinstance(root_values, list)
        and isinstance(static_values, list)
        and len(root_values) == len(moves)
        and len(static_values) == len(moves)
    ):
        hard_count = max(1, max_positions // 2)
        ranked = sorted(
            (int(ply) for ply in eligible),
            key=lambda ply: abs(float(root_values[ply]) - float(static_values[ply])),
            reverse=True,
        )
        hard = ranked[:hard_count]
        hard_set = set(hard)
        remaining = [int(ply) for ply in eligible if int(ply) not in hard_set]
        random_count = max_positions - len(hard)
        random_part = (
            game_rng.choice(remaining, size=random_count, replace=False).tolist()
            if random_count > 0
            else []
        )
        return np.array(sorted(hard + random_part), dtype=np.int32)

    return np.sort(
        game_rng.choice(eligible, size=max_positions, replace=False)
    ).astype(np.int32)


class TrainingCacheDataset(IterableDataset):
    """Read rebuildable NPZ training shards and apply augmentation on the fly."""

    def __init__(
        self,
        cache_dir,
        max_samples=2_000_000,
        symmetry_augmentation=True,
        color_swap_augmentation=True,
    ):
        self.cache_dir = cache_dir
        self.files = sorted(glob.glob(os.path.join(cache_dir, "cache_*.npz")))
        self.max_samples = max_samples
        self.symmetry_augmentation = symmetry_augmentation
        self.color_swap_augmentation = color_swap_augmentation
        override_path = os.path.join(cache_dir, "reanalysis_overrides.npz")
        self.overrides = {}
        if os.path.exists(override_path):
            with np.load(override_path) as data:
                self.overrides = {
                    int(index): float(target)
                    for index, target in zip(data["sample_index"], data["target"])
                }

    def __iter__(self):
        rng = np.random.default_rng()
        files = list(self.files)
        rng.shuffle(files)
        yielded = 0
        for path in files:
            with np.load(path) as data:
                features = data["features"]
                targets = data["targets"]
                sample_indices = data["sample_index"]
                order = rng.permutation(len(targets))
                for row_index in order:
                    feat = features[row_index].astype(np.float32, copy=True)
                    sample_index = int(sample_indices[row_index])
                    target = self.overrides.get(
                        sample_index,
                        float(targets[row_index]),
                    )
                    transform = (
                        int(rng.integers(0, SYMMETRY_COUNT))
                        if self.symmetry_augmentation
                        else SYM_IDENTITY
                    )
                    color_swap = (
                        bool(rng.integers(0, 2))
                        if self.color_swap_augmentation
                        else False
                    )
                    feat, target = augment_features(
                        feat,
                        target,
                        transform=transform,
                        color_swap=color_swap,
                    )
                    yield (
                        torch.from_numpy(feat),
                        torch.tensor([target], dtype=torch.float32),
                    )
                    yielded += 1
                    if yielded >= self.max_samples:
                        return


class DarkChessDataset(IterableDataset):
    def __init__(
        self,
        files,
        input_size=CURRENT_INPUT_SIZE,
        max_positions_per_game=4,
        max_samples=2_000_000,
        symmetry_augmentation=True,
        color_swap_augmentation=True,
    ):
        self.files = files
        self.input_size = input_size
        self.max_positions_per_game = max_positions_per_game
        self.max_samples = max_samples
        self.symmetry_augmentation = symmetry_augmentation
        self.color_swap_augmentation = color_swap_augmentation
        
    def __iter__(self):
        files = list(self.files)
        rng = np.random.default_rng()
        rng.shuffle(files)
        yielded = 0
        for f in files:
            invalid_games = 0
            try:
                with gzip.open(f, 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        if not line.strip(): continue
                        try:
                            game = json.loads(line)
                            version = str(game.get("ver", ""))
                            if not is_supported_replay_version(version):
                                # Earlier records use different draw and
                                # perpetual-chase rules, so their value targets
                                # are not valid training labels for this ruleset.
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
                            static_values = game.get('v')
                            if static_values is not None and len(static_values) != len(moves):
                                raise ValueError("v and mov lengths differ")

                            validation_board = DarkChessBoardPy(bag=game['hid'])
                            for move in moves:
                                validation_board.make_move(int(move), validate=True)
                            over, replay_result = validation_board.is_game_over()
                            if not over or replay_result != res:
                                raise ValueError(
                                    f"recorded result {res} does not match replay "
                                    f"terminal state {(over, replay_result)}"
                                )

                            eligible = select_training_plies(
                                game,
                                moves,
                                self.max_positions_per_game,
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
                                    transform = (
                                        int(rng.integers(0, SYMMETRY_COUNT))
                                        if self.symmetry_augmentation
                                        else SYM_IDENTITY
                                    )
                                    color_swap = (
                                        bool(rng.integers(0, 2))
                                        if self.color_swap_augmentation
                                        else False
                                    )
                                    feat, target = augment_features(
                                        feat,
                                        target,
                                        transform=transform,
                                        color_swap=color_swap,
                                    )
                                    yield torch.tensor(feat), torch.tensor([target], dtype=torch.float32)
                                    yielded += 1
                                    if yielded >= self.max_samples:
                                        return
                                board.make_move(int(move), validate=True)
                        except Exception as e:
                            invalid_games += 1
                            if invalid_games <= 3:
                                print(f"Skipping invalid game in {f}: {e}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
            if invalid_games:
                print(f"Skipped {invalid_games} invalid games in {f}.")

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
    symmetry_augmentation = _env_enabled("SYMMETRY_AUGMENTATION", True)
    color_swap_augmentation = _env_enabled("COLOR_SWAP_AUGMENTATION", True)
    cache_dir = os.environ.get("TRAINING_CACHE_DIR", "training_cache")
    cache_files = glob.glob(os.path.join(cache_dir, "cache_*.npz"))
    if cache_files:
        print(
            f"Using binary training cache from {cache_dir}: "
            f"{len(cache_files)} shards."
        )
        dataset = TrainingCacheDataset(
            cache_dir,
            max_samples=max_samples,
            symmetry_augmentation=symmetry_augmentation,
            color_swap_augmentation=color_swap_augmentation,
        )
    else:
        print("Binary cache not found; falling back to raw replay parsing.")
        dataset = DarkChessDataset(
            files,
            input_size=CURRENT_INPUT_SIZE,
            max_positions_per_game=max_positions,
            max_samples=max_samples,
            symmetry_augmentation=symmetry_augmentation,
            color_swap_augmentation=color_swap_augmentation,
        )
    batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
    epochs = int(os.environ.get("TRAINING_EPOCHS", "3"))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print(
        f"Hyperparameters: lr={learning_rate}, batch_size={batch_size}, "
        f"epochs={epochs}, symmetry_augmentation={symmetry_augmentation}, "
        f"color_swap_augmentation={color_swap_augmentation}"
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
