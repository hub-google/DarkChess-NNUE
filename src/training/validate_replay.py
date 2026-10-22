"""Strict standalone real-game replay validator for CPU workflow acceptance."""
import argparse
import gzip
import json
from pathlib import Path

from board import DarkChessBoardPy, NONE
from replay_format import is_supported_replay_version


def validate_record(record):
    if not is_supported_replay_version(record.get("ver", "")):
        raise ValueError("unsupported replay ruleset")
    moves = record["mov"]
    if not moves or not isinstance(moves, list):
        raise ValueError("missing moves")
    for field in ("q", "v", "qd", "pi"):
        if field in record and len(record[field]) != len(moves):
            raise ValueError(f"{field} length does not match moves")
    board = DarkChessBoardPy(bag=record["hid"])
    for ply, move in enumerate(moves):
        if "qd" in record and int(record["qd"][ply]) < 0:
            raise ValueError("negative completed search depth")
        for field in ("q", "v"):
            if field in record and not -1.0 <= float(record[field][ply]) <= 1.0:
                raise ValueError(f"{field} outside [-1,1]")
        if "pi" in record:
            distribution = record["pi"][ply]
            if not isinstance(distribution, list) or len(distribution) > 16:
                raise ValueError("invalid search policy length")
            if board.side_to_move == NONE and distribution:
                raise ValueError("the first flip has no known side")
            if distribution:
                if "qd" in record and int(record["qd"][ply]) == 0:
                    raise ValueError("fallback cannot be a search-policy label")
                legal = set(int(m) for m in board.generate_legal_moves())
                seen = set()
                total = 0.0
                for row in distribution:
                    if not isinstance(row, list) or len(row) != 2:
                        raise ValueError("malformed policy row")
                    action, probability = int(row[0]), float(row[1])
                    if action in seen or action not in legal or not 0 <= probability <= 1:
                        raise ValueError("invalid or illegal search-policy action")
                    seen.add(action)
                    total += probability
                if abs(total - 1.0) > 1e-4:
                    raise ValueError("search-policy probabilities are not normalized")
        board.make_move(int(move), validate=True)
    terminal, result = board.is_game_over()
    if not terminal or float(record["res"]) != float(result):
        raise ValueError("record is not a completed terminal game")
     return len(moves)


def validate_directory(directory):
    games = 0
    for path in sorted(Path(directory).glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    moves = validate_record(json.loads(line))
                    games += 1
                    print(f"[Replay] Valid game {games}: {moves} moves ({path.name})")
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--min-games", type=int, default=1)
    args = parser.parse_args()
    valid = validate_directory(args.directory)
    if valid < args.min_games:
        raise SystemExit(f"Expected at least {args.min_games} complete games; found {valid}")
    print(f"[Replay] VERIFIED {valid} complete games")
