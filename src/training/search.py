from dataclasses import dataclass
import math
import numpy as np

from board import (
    BLACK,
    CANNON,
    NONE,
    PIECE_COLOR,
    PIECE_TYPE,
    RED,
    decode_move,
    popcount,
)


@dataclass
class SearchResult:
    move: int
    value: float
    move_values: dict
    nodes: int


PIECE_VALUES = np.array(
    [12.0, 7.0, 6.0, 5.0, 4.0, 5.0, 2.0] * 2,
    dtype=np.float32,
)


def material_evaluate(board):
    """Red-perspective bootstrap evaluator used when no champion exists."""
    score = 0.0
    scale = 0.0
    for piece in range(14):
        value = float(PIECE_VALUES[piece])
        count = popcount(board.piece_bitboards[piece])
        # Face-down pieces still have material value, but less immediate force.
        count += 0.65 * int(board.remaining_counts[piece])
        sign = 1.0 if PIECE_COLOR[piece] == RED else -1.0
        score += sign * value * count
        scale += value * count
    if scale == 0:
        return 0.0
    return float(np.tanh(2.5 * score / scale))


def _is_flip(move):
    from_sq, to_sq, _ = decode_move(int(move))
    return from_sq == to_sq


def _ordered_moves(board, moves):
    def priority(move):
        from_sq, to_sq, is_flip = decode_move(int(move))
        if is_flip:
            return 1
        if (int(board.occupied_bitboard) >> to_sq) & 1:
            return 0
        return 2

    return sorted((int(move) for move in moves), key=priority)


class ChanceSearch:
    """
    Alpha-beta decision search with exact public-probability chance nodes.

    hidden_pieces is never read here. A flip branches over every piece type
    whose public remaining count is non-zero and weights it by count / total.
    Values are always from Red's perspective.
    """

    def __init__(self, evaluator=None, max_depth=2):
        self.evaluator = evaluator or material_evaluate
        self.max_depth = max(1, int(max_depth))
        self.nodes = 0
        self.cache = {}

    def _cache_key(self, board, depth):
        return (
            board.get_snapshot(),
            int(board.half_move_clock),
            tuple(board.history),
            depth,
        )

    def _evaluate_leaf_boards(self, boards):
        values = np.zeros(len(boards), dtype=np.float64)
        pending_indices = []
        pending_boards = []
        for index, board in enumerate(boards):
            self.nodes += 1
            over, result = board.is_game_over()
            if over:
                values[index] = float(result)
            else:
                pending_indices.append(index)
                pending_boards.append(board)

        if pending_boards:
            evaluate_many = getattr(self.evaluator, "evaluate_many", None)
            if evaluate_many is None:
                pending_values = [self.evaluator(board) for board in pending_boards]
            else:
                pending_values = evaluate_many(pending_boards)
            for index, value in zip(pending_indices, pending_values):
                values[index] = float(np.clip(value, -1.0, 1.0))
        return values

    def _analyze_depth_one(self, board):
        moves = _ordered_moves(board, board.generate_legal_moves())
        leaves = []
        leaf_metadata = []
        total = int(board.remaining_counts.sum())
        representative_flip = None

        for move in moves:
            if _is_flip(move):
                if self.evaluator is material_evaluate and representative_flip is not None:
                    continue
                representative_flip = move
                for piece, count in enumerate(board.remaining_counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    child = board.clone()
                    child.make_move(move, flip_piece=piece, validate=False)
                    leaves.append(child)
                    leaf_metadata.append((move, count / total))
            else:
                child = board.clone()
                child.make_move(move, validate=False)
                leaves.append(child)
                leaf_metadata.append((move, 1.0))

        self.nodes = 0
        leaf_values = self._evaluate_leaf_boards(leaves)
        values = {move: 0.0 for move in moves}
        for (move, weight), value in zip(leaf_metadata, leaf_values):
            values[move] += weight * float(value)
        if self.evaluator is material_evaluate and representative_flip is not None:
            for move in moves:
                if _is_flip(move):
                    values[move] = values[representative_flip]

        if board.side_to_move == RED:
            best_move = max(moves, key=lambda move: values[move])
        else:
            best_move = min(moves, key=lambda move: values[move])
        return SearchResult(best_move, float(values[best_move]), values, self.nodes)

    def _flip_value(self, board, move, depth):
        total = int(board.remaining_counts.sum())
        if total <= 0:
            raise ValueError("flip move generated with an empty bag")

        expected = 0.0
        for piece, count in enumerate(board.remaining_counts):
            count = int(count)
            if count <= 0:
                continue
            child = board.clone()
            child.make_move(move, flip_piece=piece, validate=False)
            # Do not pass decision-node bounds through a chance node: doing so
            # without Star1/Star2 bounds would be unsound.
            value = self._value(child, depth - 1, -math.inf, math.inf)
            expected += (count / total) * value
        return expected

    def _move_value(self, board, move, depth, alpha, beta):
        if _is_flip(move):
            return self._flip_value(board, move, depth)
        child = board.clone()
        child.make_move(move, validate=False)
        return self._value(child, depth - 1, alpha, beta)

    def _value(self, board, depth, alpha, beta):
        self.nodes += 1
        over, result = board.is_game_over()
        if over:
            return float(result)
        if depth <= 0:
            return float(np.clip(self.evaluator(board), -1.0, 1.0))

        key = self._cache_key(board, depth)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        moves = _ordered_moves(board, board.generate_legal_moves())
        cutoff = False
        if board.side_to_move == RED:
            best = -math.inf
            for move in moves:
                best = max(best, self._move_value(board, move, depth, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    cutoff = True
                    break
        elif board.side_to_move == BLACK:
            best = math.inf
            for move in moves:
                best = min(best, self._move_value(board, move, depth, alpha, beta))
                beta = min(beta, best)
                if alpha >= beta:
                    cutoff = True
                    break
        else:
            raise ValueError("search requires colors to be assigned by the first flip")

        best = float(np.clip(best, -1.0, 1.0))
        if not cutoff:
            self.cache[key] = best
        return best

    def analyze(self, board):
        if board.side_to_move == NONE:
            raise ValueError("the first flip must be selected before search")
        moves = _ordered_moves(board, board.generate_legal_moves())
        if not moves:
            raise ValueError("cannot search a position with no legal moves")
        if self.max_depth == 1:
            return self._analyze_depth_one(board)

        self.nodes = 0
        self.cache.clear()
        values = {}
        for move in moves:
            values[move] = self._move_value(
                board,
                move,
                self.max_depth,
                -math.inf,
                math.inf,
            )

        if board.side_to_move == RED:
            best_move = max(moves, key=lambda move: values[move])
        else:
            best_move = min(moves, key=lambda move: values[move])
        return SearchResult(
            move=best_move,
            value=float(values[best_move]),
            move_values=values,
            nodes=self.nodes,
        )

    def analyze_first_flip(self, board):
        """
        Evaluate the opening square from the first player's perspective.

        The revealed color becomes the first player's color, so a Red outcome
        uses V and a Black outcome uses -V. This keeps the opening fair without
        peeking at the actual piece under any square.
        """
        if board.side_to_move != NONE:
            raise ValueError("analyze_first_flip requires the initial position")
        moves = _ordered_moves(board, board.generate_legal_moves())
        total = int(board.remaining_counts.sum())
        if self.max_depth == 1:
            if self.evaluator is material_evaluate:
                representative = moves[0]
                expected = 0.0
                leaves = []
                weights = []
                for piece, count in enumerate(board.remaining_counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    child = board.clone()
                    child.make_move(representative, flip_piece=piece, validate=False)
                    sign = 1.0 if PIECE_COLOR[piece] == RED else -1.0
                    leaves.append(child)
                    weights.append(sign * count / total)
                self.nodes = 0
                leaf_values = self._evaluate_leaf_boards(leaves)
                for weight, value in zip(weights, leaf_values):
                    expected += weight * float(value)
                values = {move: float(expected) for move in moves}
                return SearchResult(representative, float(expected), values, self.nodes)

            leaves = []
            metadata = []
            for move in moves:
                for piece, count in enumerate(board.remaining_counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    child = board.clone()
                    child.make_move(move, flip_piece=piece, validate=False)
                    sign = 1.0 if PIECE_COLOR[piece] == RED else -1.0
                    leaves.append(child)
                    metadata.append((move, sign * count / total))

            self.nodes = 0
            leaf_values = self._evaluate_leaf_boards(leaves)
            values = {move: 0.0 for move in moves}
            for (move, weight), value in zip(metadata, leaf_values):
                values[move] += weight * float(value)
            best_move = max(moves, key=lambda move: values[move])
            return SearchResult(
                best_move,
                float(values[best_move]),
                values,
                self.nodes,
            )

        self.nodes = 0
        self.cache.clear()
        values = {}
        for move in moves:
            expected_utility = 0.0
            for piece, count in enumerate(board.remaining_counts):
                count = int(count)
                if count <= 0:
                    continue
                child = board.clone()
                child.make_move(move, flip_piece=piece, validate=False)
                red_value = self._value(
                    child,
                    self.max_depth - 1,
                    -math.inf,
                    math.inf,
                )
                first_player_value = (
                    red_value if PIECE_COLOR[piece] == RED else -red_value
                )
                expected_utility += (count / total) * first_player_value
            values[move] = float(np.clip(expected_utility, -1.0, 1.0))

        best_move = max(moves, key=lambda move: values[move])
        return SearchResult(
            move=best_move,
            value=values[best_move],
            move_values=values,
            nodes=self.nodes,
        )


def select_move(result, color, temperature=0.0, rng=None):
    if temperature <= 0 or len(result.move_values) == 1:
        return result.move

    rng = rng or np.random.default_rng()
    moves = np.array(list(result.move_values.keys()), dtype=np.int32)
    red_values = np.array(
        [result.move_values[int(move)] for move in moves],
        dtype=np.float64,
    )
    utilities = red_values if color == RED else -red_values
    logits = utilities / max(float(temperature), 1e-6)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(moves, p=probabilities))


def select_first_flip(result, temperature=0.0, rng=None):
    if temperature <= 0 or len(result.move_values) == 1:
        return result.move
    rng = rng or np.random.default_rng()
    moves = np.array(list(result.move_values.keys()), dtype=np.int32)
    utilities = np.array(
        [result.move_values[int(move)] for move in moves],
        dtype=np.float64,
    )
    logits = utilities / max(float(temperature), 1e-6)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(moves, p=probabilities))
