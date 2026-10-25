"""Independent Rust MistyBanqi UCI adapter, with public-state-only Banqi FEN.

External engine source is separately licensed MIT. Its own rules may differ
from the project's perpetual-chase rules, so legal root moves are restricted
to the host arbiter's moves. The opponent NEVER receives the hidden deal.
"""
import subprocess

from board import BLACK, NONE, RED, decode_move

LETTERS = "GAERHCSgaerhcs"


def move_to_uci(move):
    source, target, _ = decode_move(int(move))
    def square(idx):
        return f"{chr(ord('a') + idx % 8)}{idx // 8}"
    return square(source) + square(target)


def uci_to_move(s):
    if len(s) != 4 or not all(
        "a" <= s[i] <= "h" and "0" <= s[i + 1] <= "3" for i in (0, 2)
    ):
        raise ValueError(f"invalid UCI Banqi move: {s!r}")
    source = ord(s[0]) - 97 + 8 * int(s[1])
    target = ord(s[2]) - 97 + 8 * int(s[3])
    return int((source << 5) | target)


def public_fen(board):
    """No private hidden_pieces reads. Square 0 = a0, bottom FEN rank."""
    squares = [""] * 32
    for square in range(32):
        if int(board.hidden_bitboard) & (1 << square):
            squares[square] = "X"
            continue
        for piece, bitboard in enumerate(board.piece_bitboards):
            if int(bitboard) & (1 << square):
                squares[square] = LETTERS[piece]
                break
        else:
            squares[square] = "1"
    ranks = []
    for row in range(3, -1, -1):
        rank = ""
        empty = 0
        for s in squares[row * 8:(row + 1) * 8]:
            if s == "1":
                empty += 1
            else:
                if empty:
                    rank += str(empty)
                    empty = 0
                rank += s
        if empty:
            rank += str(empty)
        ranks.append(rank)
    pool = "".join(
        LETTERS[piece] + str(int(count))
        for piece, count in enumerate(board.remaining_counts) if count > 0
    ) or "-"
    color = "r" if board.side_to_move == RED else ("b" if board.side_to_move == BLACK else "-")
    return f"{'/'.join(ranks)} {color} {pool} {board.half_move_clock} 1"


def choose_misty_move(board, executable, node_budget=500, max_seconds=0.15):
    legal = [int(move) for move in board.generate_legal_moves()]
    if not legal:
        raise ValueError("cannot ask an external engine to play terminal state")
    if node_budget < 1 or max_seconds <= 0:
        raise ValueError("invalid search budget")
    allowed = " ".join(move_to_uci(move) for move in legal)
    script = (
        "uci\n"
        "setoption name NoProgressLimit value 60\n"
        "isready\n"
        f"position fen {public_fen(board)}\n"
        f"go nodes {int(node_budget)} movetime {max(1, int(max_seconds * 1000))} searchmoves {allowed}\n"
        "quit\n"
    )
    try:
        result = subprocess.run(
            [executable], input=script, text=True, capture_output=True,
            check=True, timeout=max(5, max_seconds + 3),
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError("external MistyBanqi engine did not return a valid decision") from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.startswith("bestmove ")]
    if len(lines) != 1:
        raise RuntimeError(f"missing/duplicate UCI bestmove (output={result.stdout[-400:]!r})")
    proposed = uci_to_move(lines[0].split()[1])
    if proposed not in legal:
        raise RuntimeError(f"MistyBanqi returned illegal move {lines[0]}: incompatible game rules?")
    return proposed
