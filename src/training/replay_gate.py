"""Machine-readable readiness gate for autonomous GitHub training cycles.

Count completed JSONL entries in a gzip replay snapshot without constructing
large NPZ shards or running expensive reanalysis. This gate never promotes a
model; the strict cache manifest validation remains mandatory before training.
"""
import argparse
import gzip
from pathlib import Path


def count_completed_replays(filename, minimum):
    if int(minimum) < 1:
        raise ValueError("minimum must be positive")
    count = 0
    with gzip.open(filename, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count, count >= int(minimum)


def write_github_output(filename, count, ready):
    if filename is None:
        return
    with open(filename, "a", encoding="utf-8") as stream:
        stream.write(f"ready={'true' if ready else 'false'}\n")
        stream.write(f"replay_games={count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--minimum", type=int, default=1000)
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args()
    count, ready = count_completed_replays(args.replay, args.minimum)
    write_github_output(args.github_output, count, ready)
    print(f"[Replay Gate] games={count} minimum={args.minimum} ready={ready}")
    if not ready:
        print("[Replay Gate] Keep accumulating; this is a successful no-train cycle.")
