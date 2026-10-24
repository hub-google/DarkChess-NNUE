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


def replay_ids(path: str) -> set[str]:
    """Return immutable replay IDs so staged retries are not double-counted."""
    ids = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                game_id = json.loads(line)["id"]
                if not isinstance(game_id, str) or not game_id:
                    raise ValueError(f"Invalid replay ID in {path}")
                ids.add(game_id)
    return ids


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
    consolidated_ids: set[str] = set()
    pending_ids: set[str] = set()
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename="replay_buffer.jsonl.gz", token=token,
                local_dir=temp_dir,
            )
            consolidated_ids = replay_ids(local)
        except EntryNotFoundError:
            print("[Training Status] No consolidated replay buffer yet.")

        def read_batch(name):
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename=name, token=token, local_dir=temp_dir,
            )
            return replay_ids(local)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for batch_ids in pool.map(read_batch, staged):
                pending_ids.update(batch_ids)
        print(f"[Training Status] Counted {len(staged)} staging files.")

    # A verified replay buffer may temporarily coexist with staging sources
    # while their deletion is retried. Those replays count exactly once.
    summary = {
        "replayGames": len(consolidated_ids),
        "stagingGames": len(pending_ids - consolidated_ids),
        "totalGames": len(consolidated_ids | pending_ids),
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
