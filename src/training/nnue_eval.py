import numpy as np
import torch

from board import BLACK, INITIAL_COUNTS, RED, decode_move
from train import CURRENT_INPUT_SIZE, LEGACY_INPUT_SIZE, extract_features


def _global_feature_values(board, input_size):
    if input_size == LEGACY_INPUT_SIZE:
        return {
            480 + piece: float(board.remaining_counts[piece])
            for piece in range(14)
        }

    values = {
        480 + piece: (
            float(board.remaining_counts[piece]) / float(INITIAL_COUNTS[piece])
        )
        for piece in range(14)
    }
    values[494] = 1.0 if board.side_to_move == RED else 0.0
    values[495] = 1.0 if board.side_to_move == BLACK else 0.0
    values[496] = min(float(board.half_move_clock) / 60.0, 1.0)
    values[497] = min(float(board.repetition_count()) / 3.0, 1.0)
    return values


def feature_deltas(
    parent,
    child,
    move,
    flip_piece=None,
    input_size=CURRENT_INPUT_SIZE,
):
    """Return sparse (feature_index, delta) updates from parent to child."""
    from_sq, to_sq, is_flip = decode_move(int(move))
    deltas = []

    if is_flip:
        if flip_piece is None:
            flip_piece = child._piece_at(from_sq)
        if flip_piece < 0:
            raise ValueError("could not identify flipped piece")
        deltas.append((from_sq * 15 + 14, -1.0))
        deltas.append((from_sq * 15 + int(flip_piece), 1.0))
    else:
        attacker = parent._piece_at(from_sq)
        if attacker < 0:
            raise ValueError("could not identify moving piece")
        victim = parent._piece_at(to_sq)
        deltas.append((from_sq * 15 + attacker, -1.0))
        if victim >= 0:
            deltas.append((to_sq * 15 + victim, -1.0))
        deltas.append((to_sq * 15 + attacker, 1.0))

    parent_globals = _global_feature_values(parent, input_size)
    child_globals = _global_feature_values(child, input_size)
    for index, old_value in parent_globals.items():
        delta = child_globals[index] - old_value
        if delta:
            deltas.append((index, float(delta)))

    return deltas


class ModelEvaluator:
    """
    CPU NNUE evaluator with an incremental first-layer accumulator.

    Search calls prepare_child after making a move. The child accumulator is
    derived from the parent by adding only changed feature columns, avoiding a
    full 498x256 first-layer matrix multiply at every evaluated leaf.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.input_size = int(model.input_size)
        self._token = object()

    def _cache(self, board):
        cache = getattr(board, "_nnue_accumulators", None)
        if cache is None:
            cache = {}
            board._nnue_accumulators = cache
        return cache

    def _full_accumulator(self, board):
        features = extract_features(board, self.input_size)
        tensor = torch.from_numpy(features)
        with torch.no_grad():
            return self.model.fc1(tensor).detach()

    def accumulator(self, board):
        cache = self._cache(board)
        accumulator = cache.get(self._token)
        if accumulator is None:
            accumulator = self._full_accumulator(board)
            cache[self._token] = accumulator
        return accumulator

    def prepare_child(self, parent, child, move, flip_piece=None):
        parent_acc = self.accumulator(parent)
        child_acc = parent_acc.clone()
        with torch.no_grad():
            for feature_index, delta in feature_deltas(
                parent,
                child,
                move,
                flip_piece=flip_piece,
                input_size=self.input_size,
            ):
                child_acc.add_(
                    self.model.fc1.weight[:, int(feature_index)],
                    alpha=float(delta),
                )
        self._cache(child)[self._token] = child_acc
        return child_acc

    def _forward_from_accumulator(self, accumulator):
        with torch.no_grad():
            x = torch.clamp(torch.relu(accumulator), max=1.0)
            x = torch.clamp(torch.relu(self.model.fc2(x)), max=1.0)
            return torch.tanh(self.model.fc3(x))

    def __call__(self, board):
        return float(
            self._forward_from_accumulator(self.accumulator(board)).item()
        )

    def evaluate_many(self, boards):
        if not boards:
            return np.zeros(0, dtype=np.float32)
        accumulators = torch.stack(
            [self.accumulator(board) for board in boards]
        )
        with torch.no_grad():
            x = torch.clamp(torch.relu(accumulators), max=1.0)
            x = torch.clamp(torch.relu(self.model.fc2(x)), max=1.0)
            return torch.tanh(self.model.fc3(x)).squeeze(1).cpu().numpy()
