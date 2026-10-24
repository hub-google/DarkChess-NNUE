"""Publish a safe, public game-count summary without exposing the HF token."""

import gzip
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "hub-google/DarkChess-NNUE-Data"


def count_lines(path: str) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    # Listing the repo root also walks giant archival subtrees. List only the
    # live staging folder and the known canonical replay path.
    try:
        staged = [
            entry.path for entry in api.list_repo_tree(
                repo_id=REPO_ID, path_in_repo="staging",
                recursive=True, repo_type="dataset",
            ) if getattr(entry, "path", "").endswith(".jsonl.gz")
        ]
    except EntryNotFoundError:
        staged = []
        print("[Training Status] No staging batches yet.")
    replay_games = 0
    staging_games = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename="replay_buffer.jsonl.gz", token=token,
                local_dir=temp_dir,
            )
            replay_games = count_lines(local)
        except EntryNotFoundError:
            print("[Training Status] No consolidated replay buffer yet.")

        def read_batch(name):
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename=name, token=token, local_dir=temp_dir,
            )
            return count_lines(local)

        with ThreadPoolExecutor(max_workers=6) as pool:
            staging_games = sum(pool.map(read_batch, staged))
        print(f"[Training Status] Counted {len(staged)} staging files.")

    summary = {
        "replayGames": replay_games,
        "stagingGames": staging_games,
        "totalGames": replay_games + staging_games,
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output = Path("frontend/public/training-status.json")
    if output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
            if all(previous.get(key) == summary[key] for key in ("replayGames", "stagingGames", "totalGames")):
                print("[Training Status] Counts unchanged; no Pages redeployment needed.")
                return
        except (ValueError, TypeError):
            pass
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Published training summary: {summary['totalGames']} games")


if __name__ == "__main__":
    main()
