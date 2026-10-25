from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from board import BLACK, RED, DarkChessBoardPy, encode_move


BOARD_ROWS = 4
BOARD_COLS = 8
SYMMETRIES = (0, 1, 2, 3)


class TablebaseLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class TablebaseResult:
    value: float
    distance: int
    best_move: int | None = None
    states: int = 0


def transform_square(square, symmetry):
    row, col = divmod(int(square), BOARD_COLS)
    if symmetry in (1, 3):
        col = BOARD_COLS - 1 - col
    if symmetry in (2, 3):
        row = BOARD_ROWS - 1 - row
    return row * BOARD_COLS + col


def transform_bitboard(bitboard, symmetry):
    source = int(bitboard)
    result = 0
    while source:
        lsb = source & -source
        square = lsb.bit_length() - 1
        result |= 1 << transform_square(square, symmetry)
        source ^= lsb
    return result


def _transform_token(token, symmetry):
    token = int(token)
    return -1 if token < 0 else transform_square(token, symmetry)


def _transform_chase(entry, symmetry):
    if entry is None:
        return None
    attacker_token, target_token, count, chaser, route = entry
    return (
        _transform_token(attacker_token, symmetry),
        _transform_token(target_token, symmetry),
        int(count),
        int(chaser),
        tuple(transform_square(square, symmetry) for square in route),
    )


def canonical_key(board):
    """Canonicalize a fully revealed endgame under the four valid symmetries."""
    if int(board.hidden_bitboard) != 0 or int(board.remaining_counts.sum()) != 0:
        raise ValueError("tablebase requires a fully revealed position")

    candidates = []
    for symmetry in SYMMETRIES:
        bitboards = tuple(
            transform_bitboard(bb, symmetry)
            for bb in board.piece_bitboards
        )
        tokens = [-1] * 32
        for square, token in enumerate(board.token_at_square):
            mapped_square = transform_square(square, symmetry)
            tokens[mapped_square] = _transform_token(token, symmetry)
        chase_threats = tuple(
            sorted(
                _transform_chase(entry, symmetry)
                for entry in board.chase_threats
            )
        )
        pending_chase = _transform_chase(board.pending_chase, symmetry)
        candidates.append(
            (
                bitboards,
                int(board.side_to_move),
                int(board.half_move_clock),
                tuple(tokens),
                chase_threats,
                pending_chase,
            )
        )
    return min(candidates)


def visible_piece_count(board):
    return sum(int(bb).bit_count() for bb in board.piece_bitboards)


def _ordered_moves(board):
    """Captures first; immediate forced wins can terminate exact minimax early."""
    def priority(move):
        to_sq = int(move) & 31
        from_sq = (int(move) >> 5) & 31
        is_flip = from_sq == to_sq
        is_capture = (
            not is_flip
            and ((int(board.occupied_bitboard) >> to_sq) & 1)
        )
        return 0 if is_capture else 1

    return sorted(
        (int(move) for move in board.generate_legal_moves()),
        key=priority,
    )


def _prefer(candidate, incumbent, side):
    if incumbent is None:
        return True
    cand_value, cand_distance = candidate
    inc_value, inc_distance = incumbent

    if side == RED:
        if cand_value != inc_value:
            return cand_value > inc_value
        if cand_value > 0:
            return cand_distance < inc_distance
        if cand_value < 0:
            return cand_distance > inc_distance
        return cand_distance < inc_distance

    if side == BLACK:
        if cand_value != inc_value:
            return cand_value < inc_value
        if cand_value < 0:
            return cand_distance < inc_distance
        if cand_value > 0:
            return cand_distance > inc_distance
        return cand_distance < inc_distance

    raise ValueError("tablebase position must have a side to move")


class EndgameTablebase:
    """
    Exact on-demand W/D/L solver for fully revealed small endgames.

    It uses the production board move generator, capture rules, half-move draw
    rule and perpetual-chase state. Canonicalization collapses horizontal,
    vertical and 180-degree symmetric states. max_states is a safety fuse:
    a query is exact if it completes, otherwise it raises rather than silently
    returning a heuristic value.
    """

    def __init__(self, max_pieces=4, max_states=2_000_000):
        self.max_pieces = int(max_pieces)
        self.max_states = int(max_states)
        self.cache = {}
        self.states = 0

    def clear(self):
        self.cache.clear()
        self.states = 0

    def _validate(self, board):
        if board.side_to_move not in (RED, BLACK):
            raise ValueError("tablebase requires an assigned side to move")
        if int(board.hidden_bitboard) != 0 or int(board.remaining_counts.sum()) != 0:
            raise ValueError("tablebase requires no hidden pieces")
        pieces = visible_piece_count(board)
        if pieces > self.max_pieces:
            raise ValueError(
                f"tablebase supports at most {self.max_pieces} visible pieces; got {pieces}"
            )

    def _solve(self, board):
        key = canonical_key(board)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.states >= self.max_states:
            raise TablebaseLimitExceeded(
                f"tablebase exceeded {self.max_states} canonical states"
            )
        self.states += 1

        over, result = board.is_game_over()
        if over:
            answer = (float(result), 0)
            self.cache[key] = answer
            return answer

        moves = _ordered_moves(board)
        if not moves:
            value = -1.0 if board.side_to_move == RED else 1.0
            answer = (value, 0)
            self.cache[key] = answer
            return answer

        best = None
        side = int(board.side_to_move)
        for move in moves:
            child = board.clone()
            child.make_move(move, validate=False)
            value, distance = self._solve(child)
            candidate = (value, distance + 1)
            if _prefer(candidate, best, side):
                best = candidate
                if (
                    (side == RED and best == (1.0, 1))
                    or (side == BLACK and best == (-1.0, 1))
                ):
                    break

        self.cache[key] = best
        return best

    def probe(self, board):
        self._validate(board)
        value, distance = self._solve(board)
        return TablebaseResult(
            value=float(value),
            distance=int(distance),
            states=self.states,
        )

    def analyze(self, board):
        self._validate(board)
        over, result = board.is_game_over()
        if over:
            return TablebaseResult(
                value=float(result),
                distance=0,
                best_move=None,
                states=self.states,
            )

        side = int(board.side_to_move)
        best = None
        best_move = None
        for move in _ordered_moves(board):
            child = board.clone()
            child.make_move(move, validate=False)
            value, distance = self._solve(child)
            candidate = (value, distance + 1)
            if _prefer(candidate, best, side):
                best = candidate
                best_move = move
                if (
                    (side == RED and best == (1.0, 1))
                    or (side == BLACK and best == (-1.0, 1))
                ):
                    break

        return TablebaseResult(
            value=float(best[0]),
            distance=int(best[1]),
            best_move=best_move,
            states=self.states,
        )


def _piece_multisets(total, piece=0, prefix=()):
    initial_counts = (1, 2, 2, 2, 2, 2, 5, 1, 2, 2, 2, 2, 2, 5)
    if piece == len(initial_counts):
        if total == 0:
            yield prefix
        return
    for count in range(min(initial_counts[piece], total) + 1):
        yield from _piece_multisets(
            total - count,
            piece + 1,
            prefix + (count,),
        )


@lru_cache(maxsize=None)
def estimate_raw_states(piece_count):
    """
    Count fully revealed board placements respecting the real inventory, times
    two sides-to-move. This excludes half-move/chase state, so it is a lower
    bound for a fully materialized rules-complete tablebase.
    """
    n = int(piece_count)
    if n < 0 or n > 32:
        raise ValueError("piece_count must be between 0 and 32")
    falling = math.factorial(32) // math.factorial(32 - n)
    placements = 0
    for counts in _piece_multisets(n):
        denominator = 1
        for count in counts:
            denominator *= math.factorial(count)
        placements += falling // denominator
    return 2 * placements


def make_test_board(pieces, side_to_move, half_move_clock=0):
    """Utility for exact tests and benchmark positions."""
    board = DarkChessBoardPy()
    board.piece_bitboards[:] = 0
    board.hidden_bitboard = np.uint32(0)
    board.occupied_bitboard = np.uint32(0)
    board.remaining_counts[:] = 0
    board.side_to_move = int(side_to_move)
    board.half_move_clock = int(half_move_clock)
    board.history = []
    board.token_at_square[:] = -1
    board.chase_threats = []
    board.pending_chase = None

    for piece, square in pieces:
        piece = int(piece)
        square = int(square)
        board.piece_bitboards[piece] |= np.uint32(1 << square)
        board.occupied_bitboard |= np.uint32(1 << square)
        board.token_at_square[square] = square
    return board


if __name__ == "__main__":
    for pieces in range(2, 7):
        states = estimate_raw_states(pieces)
        print(
            f"{pieces} pieces: raw placement lower bound={states:,}; "
            f"ideal 4-way symmetry floor~{states // 4:,}"
        )
