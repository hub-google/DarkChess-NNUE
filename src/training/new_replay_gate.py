"""Train only when a verified HF replay snapshot contains new game IDs.

The marker records the last replay set that completed model evaluation (SPRT).
The initial marker has a known count from the 2026-10-27 successful run.
Never treat a scheduled job or an unchanged old buffer as fresh training data.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def snapshot_replays(path):
    ids = set()
    newest = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            game = json.loads(line)
            game_id = game.get("id")
            if not isinstance(game_id, str) or not game_id:
                raise ValueError("Replay record has no valid game ID")
            if game_id in ids:
                raise ValueError(f"Duplicate replay game ID: {game_id}")
            ids.add(game_id)
            newest = max(newest, int(game.get("ts") or 0))
    if not ids:
        raise ValueError("Replay buffer is empty")
    digest = hashlib.sha256()
    for game_id in sorted(ids):
        digest.update(game_id.encode("utf-8"))
        digest.update(b"\n")
    return {"total_games": len(ids), "ids_sha256": digest.hexdigest(),
            "newest_ts": newest}


def has_new_games(current, previous):
    if not previous:
        raise RuntimeError("Missing last-training marker: fail closed to prevent redundant training")
    count = int(previous["total_games"])
    if current["total_games"] > count:
        return True
    if previous.get("ids_sha256") and current["ids_sha256"] != previous["ids_sha256"]:
        # New game IDs also count when the 500k rolling buffer stays at its cap.
        # Reject pure data loss: this is not a new training source.
        return current["total_games"] >= count or (
            previous.get("newest_ts") is not None
            and current["newest_ts"] > int(previous["newest_ts"])
        )
    if previous.get("newest_ts") is not None:
        return current["newest_ts"] > int(previous["newest_ts"])
    # Bootstrap marker has the known previously trained count (17230).
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--github-output")
    args = parser.parse_args()

    previous = json.loads(Path(args.marker).read_text(encoding="utf-8"))
    current = snapshot_replays(args.replay)
    fresh = has_new_games(current, previous)
    Path(args.candidate).write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"new={'true' if fresh else 'false'}\n")
    print(
        f"[New Replay Gate] previous={previous['total_games']} "
        f"current={current['total_games']} new={'true' if fresh else 'false'}"
    )
    if not fresh:
        print("[New Replay Gate] SKIP: no verified new replay games since last model evaluation.")


if __name__ == "__main__":
    main()
