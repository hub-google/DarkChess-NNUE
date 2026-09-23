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
    depth: int = 0


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


def _ordered_moves(board, moves, preferred=None):
    def priority(move):
        move = int(move)
        from_sq, to_sq, is_flip = decode_move(move)
        preferred_rank = 0 if preferred is not None and move == int(preferred) else 1
        if is_flip:
            tactical_rank = 1
        elif (int(board.occupied_bitboard) >> to_sq) & 1:
            tactical_rank = 0
        else:
            tactical_rank = 2
        return preferred_rank, tactical_rank

    return sorted((int(move) for move in moves), key=priority)


BOARD_ROWS = 4
BOARD_COLS = 8


def _opening_square_orbit(square):
    """Return the four geometric symmetries of a square on the 4x8 board."""
    row, col = divmod(int(square), BOARD_COLS)
    return tuple(sorted({
        row * BOARD_COLS + col,
        row * BOARD_COLS + (BOARD_COLS - 1 - col),
        (BOARD_ROWS - 1 - row) * BOARD_COLS + col,
        (BOARD_ROWS - 1 - row) * BOARD_COLS + (BOARD_COLS - 1 - col),
    }))


def _opening_symmetry_groups(moves):
    """
    Collapse the fully hidden opening from 32 flip squares to 8 geometric
    equivalence classes. The true game value is invariant under horizontal/
    vertical reflection and 180-degree rotation.
    """
    by_square = {}
    for move in moves:
        from_sq, to_sq, is_flip = decode_move(int(move))
        if not is_flip or from_sq != to_sq:
            return [[int(move)] for move in moves]
        by_square[from_sq] = int(move)

    groups = []
    visited = set()
    for square in sorted(by_square):
        if square in visited:
            continue
        orbit = _opening_square_orbit(square)
        group = [by_square[sq] for sq in orbit if sq in by_square]
        visited.update(orbit)
        groups.append(sorted(group))
    return groups


def _expand_group_values(groups, representative_values):
    values = {}
    for group in groups:
        value = float(representative_values[group[0]])
        for move in group:
            values[move] = value
    return values


class SearchBudgetExceeded(RuntimeError):
    pass


TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class ChanceSearch:
    """CPU-oriented expectiminimax with Star1 bounds, TT and node budgets."""

    def __init__(self, evaluator=None, max_depth=2, node_budget=None):
        self.evaluator = evaluator or material_evaluate
        self.max_depth = max(1, int(max_depth))
        self.node_budget = None if node_budget is None else max(1, int(node_budget))
        self.nodes = 0
        self.cache = {}
        self.chance_cutoffs = 0
        self._active_evaluator = self.evaluator
        self._prepare_child_hook = getattr(
            self._active_evaluator,
            "prepare_child",
            None,
        )

    def _cache_key(self, board):
        return (
            board.get_snapshot(),
            int(board.half_move_clock),
            tuple(board.history),
            tuple(board.chase_threats),
            board.pending_chase,
        )

    def _tick(self, count=1):
        self.nodes += int(count)
        if self.node_budget is not None and self.nodes > self.node_budget:
            raise SearchBudgetExceeded

    def _bind_evaluator(self, board):
        binder = getattr(self.evaluator, "for_position", None)
        self._active_evaluator = (
            binder(board) if binder is not None else self.evaluator
        )
        self._prepare_child_hook = getattr(
            self._active_evaluator,
            "prepare_child",
            None,
        )


    def _prepare_child(self, parent, child, move, flip_piece=None):
        if self._prepare_child_hook is not None:
            self._prepare_child_hook(
                parent,
                child,
                move,
                flip_piece=flip_piece,
            )

    def _evaluate_leaf_boards(self, boards):
        values = np.zeros(len(boards), dtype=np.float64)
        pending_indices = []
        pending_boards = []
        for index, board in enumerate(boards):
            self._tick()
            over, result = board.is_game_over()
            if over:
                values[index] = float(result)
            else:
                pending_indices.append(index)
                pending_boards.append(board)

        if pending_boards:
            evaluate_many = getattr(self._active_evaluator, "evaluate_many", None)
            if evaluate_many is None:
                pending_values = [
                    self._active_evaluator(board)
                    for board in pending_boards
                ]
            else:
                pending_values = evaluate_many(pending_boards)
            for index, value in zip(pending_indices, pending_values):
                values[index] = float(np.clip(value, -1.0, 1.0))
        return values

    def _analyze_depth_one(self, board):
        moves = _ordered_moves(board, board.generate_legal_moves())
        leaves = []
        metadata = []
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
                    self._prepare_child(board, child, move, flip_piece=piece)
                    leaves.append(child)
                    metadata.append((move, count / total))
            else:
                child = board.clone()
                child.make_move(move, validate=False)
                self._prepare_child(board, child, move)
                leaves.append(child)
                metadata.append((move, 1.0))

        leaf_values = self._evaluate_leaf_boards(leaves)
        values = {move: 0.0 for move in moves}
        for (move, weight), value in zip(metadata, leaf_values):
            values[move] += weight * float(value)
        if self.evaluator is material_evaluate and representative_flip is not None:
            for move in moves:
                if _is_flip(move):
                    values[move] = values[representative_flip]

        best_move = (
            max(moves, key=lambda move: values[move])
            if board.side_to_move == RED
            else min(moves, key=lambda move: values[move])
        )
        return SearchResult(best_move, float(values[best_move]), values, self.nodes, 1)

    def _flip_value(self, board, move, depth, alpha, beta):
        total = int(board.remaining_counts.sum())
        if total <= 0:
            raise ValueError("flip move generated with an empty bag")

        outcomes = sorted(
            (
                (int(count), int(piece))
                for piece, count in enumerate(board.remaining_counts)
                if int(count) > 0
            ),
            reverse=True,
        )
        expected = 0.0
        probability_done = 0.0

        for count, piece in outcomes:
            probability = count / total
            child = board.clone()
            child.make_move(move, flip_piece=piece, validate=False)
            self._prepare_child(board, child, move, flip_piece=piece)
            value = self._value(child, depth - 1, -math.inf, math.inf)
            expected += probability * value
            probability_done += probability

            remaining = max(0.0, 1.0 - probability_done)
            lower = expected - remaining
            upper = expected + remaining
            if upper <= alpha:
                self.chance_cutoffs += 1
                return float(np.clip(upper, -1.0, 1.0))
            if lower >= beta:
                self.chance_cutoffs += 1
                return float(np.clip(lower, -1.0, 1.0))

        return float(np.clip(expected, -1.0, 1.0))

    def _move_value(self, board, move, depth, alpha, beta):
        if _is_flip(move):
            return self._flip_value(board, move, depth, alpha, beta)
        child = board.clone()
        child.make_move(move, validate=False)
        self._prepare_child(board, child, move)
        return self._value(child, depth - 1, alpha, beta)

    def _value(self, board, depth, alpha, beta):
        self._tick()
        over, result = board.is_game_over()
        if over:
            return float(result)
        if depth <= 0:
            return float(np.clip(self._active_evaluator(board), -1.0, 1.0))

        key = self._cache_key(board)
        alpha0, beta0 = alpha, beta
        preferred = None
        cached = self.cache.get(key)
        if cached is not None:
            cached_depth, cached_value, flag, preferred = cached
            if cached_depth >= depth:
                if flag == TT_EXACT:
                    return cached_value
                if flag == TT_LOWER:
                    alpha = max(alpha, cached_value)
                else:
                    beta = min(beta, cached_value)
                if alpha >= beta:
                    return cached_value

        moves = _ordered_moves(board, board.generate_legal_moves(), preferred=preferred)
        best_move = None
        if board.side_to_move == RED:
            best = -math.inf
            for move in moves:
                value = self._move_value(board, move, depth, alpha, beta)
                if value > best:
                    best, best_move = value, move
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        elif board.side_to_move == BLACK:
            best = math.inf
            for move in moves:
                value = self._move_value(board, move, depth, alpha, beta)
                if value < best:
                    best, best_move = value, move
                beta = min(beta, best)
                if alpha >= beta:
                    break
        else:
            raise ValueError("search requires colors to be assigned by the first flip")

        best = float(np.clip(best, -1.0, 1.0))
        flag = TT_EXACT
        if best <= alpha0:
            flag = TT_UPPER
        elif best >= beta0:
            flag = TT_LOWER
        old = self.cache.get(key)
        if old is None or depth >= old[0]:
            self.cache[key] = (depth, best, flag, best_move)
        return best

    def _analyze_fixed(self, board, depth):
        moves = _ordered_moves(board, board.generate_legal_moves())
        if not moves:
            raise ValueError("cannot search a position with no legal moves")
        if depth == 1:
            return self._analyze_depth_one(board)

        values = {}
        for move in moves:
            values[move] = self._move_value(board, move, depth, -math.inf, math.inf)

        best_move = (
            max(moves, key=lambda move: values[move])
            if board.side_to_move == RED
            else min(moves, key=lambda move: values[move])
        )
        return SearchResult(
            move=best_move,
            value=float(values[best_move]),
            move_values=values,
            nodes=self.nodes,
            depth=depth,
        )

    def analyze(self, board):
        if board.side_to_move == NONE:
            raise ValueError("the first flip must be selected before search")

        self.nodes = 0
        self.chance_cutoffs = 0
        self.cache.clear()
        self._bind_evaluator(board)
        if self.node_budget is None:
            return self._analyze_fixed(board, self.max_depth)

        best_completed = None
        for depth in range(1, self.max_depth + 1):
            try:
                best_completed = self._analyze_fixed(board, depth)
            except SearchBudgetExceeded:
                break

        if best_completed is None:
            saved_budget = self.node_budget
            self.node_budget = None
            self.nodes = 0
            try:
                best_completed = self._analyze_fixed(board, 1)
            finally:
                self.node_budget = saved_budget

        best_completed.nodes = self.nodes
        return best_completed

    def analyze_first_flip(self, board):
        if board.side_to_move != NONE:
            raise ValueError("analyze_first_flip requires the initial position")
        self.nodes = 0
        self.chance_cutoffs = 0
        self.cache.clear()
        self._bind_evaluator(board)
        moves = _ordered_moves(board, board.generate_legal_moves())
        groups = _opening_symmetry_groups(moves)
        search_moves = [group[0] for group in groups]
        total = int(board.remaining_counts.sum())

        if self.max_depth == 1:
            if self.evaluator is material_evaluate:
                representative = search_moves[0]
                expected = 0.0
                leaves = []
                weights = []
                for piece, count in enumerate(board.remaining_counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    child = board.clone()
                    child.make_move(representative, flip_piece=piece, validate=False)
                    self._prepare_child(board, child, representative, flip_piece=piece)
                    sign = 1.0 if PIECE_COLOR[piece] == RED else -1.0
                    leaves.append(child)
                    weights.append(sign * count / total)
                values_array = self._evaluate_leaf_boards(leaves)
                for weight, value in zip(weights, values_array):
                    expected += weight * float(value)
                values = {move: float(expected) for move in moves}
                return SearchResult(representative, float(expected), values, self.nodes, 1)

            leaves = []
            metadata = []
            for move in search_moves:
                for piece, count in enumerate(board.remaining_counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    child = board.clone()
                    child.make_move(move, flip_piece=piece, validate=False)
                    self._prepare_child(board, child, move, flip_piece=piece)
                    sign = 1.0 if PIECE_COLOR[piece] == RED else -1.0
                    leaves.append(child)
                    metadata.append((move, sign * count / total))
            leaf_values = self._evaluate_leaf_boards(leaves)
            representative_values = {move: 0.0 for move in search_moves}
            for (move, weight), value in zip(metadata, leaf_values):
                representative_values[move] += weight * float(value)
            values = _expand_group_values(groups, representative_values)
            best_move = max(moves, key=lambda move: values[move])
            return SearchResult(best_move, float(values[best_move]), values, self.nodes, 1)

        representative_values = {}
        for move in search_moves:
            expected_utility = 0.0
            for piece, count in enumerate(board.remaining_counts):
                count = int(count)
                if count <= 0:
                    continue
                child = board.clone()
                child.make_move(move, flip_piece=piece, validate=False)
                self._prepare_child(board, child, move, flip_piece=piece)
                red_value = self._value(child, self.max_depth - 1, -math.inf, math.inf)
                first_player_value = red_value if PIECE_COLOR[piece] == RED else -red_value
                expected_utility += (count / total) * first_player_value
            representative_values[move] = float(np.clip(expected_utility, -1.0, 1.0))

        values = _expand_group_values(groups, representative_values)
        best_move = max(moves, key=lambda move: values[move])
        return SearchResult(
            best_move,
            values[best_move],
            values,
            self.nodes,
            self.max_depth,
        )

class ExactChanceSearch(ChanceSearch):
    """Reference expectiminimax that never applies Star1 chance cutoffs."""

    def _flip_value(self, board, move, depth, alpha, beta):
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
            self._prepare_child(board, child, move, flip_piece=piece)
            value = self._value(child, depth - 1, -math.inf, math.inf)
            expected += (count / total) * value
        return float(np.clip(expected, -1.0, 1.0))


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
