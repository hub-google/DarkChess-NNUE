"""Strict standalone real-game replay validator for CPU workflow acceptance."""
import argparse
import gzip
import json
from pathlib import Path

from board import DarkChessBoardPy
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
    for move in moves:
        board.make_move(int(move), validate=True)
    terminal, result = board.is_game_over()
    if not terminal or float(record["res"]) != float(result):
        raise ValueError("record is not a completed terminal game")
    for distribution, depth in zip(record.get("pi", []), record.get("qd", [])):
        if depth <= 0 and distribution:
            raise ValueError("fallback cannot be a search-policy label")
        if distribution and abs(sum(float(row[1]) for row in distribution) - 1) > 1e-4:
            raise ValueError("invalid search policy distribution")
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
