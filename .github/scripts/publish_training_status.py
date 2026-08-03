"""Publish a safe, public game-count summary without exposing the HF token."""

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "hub-google/DarkChess-NNUE-Data"


def count_lines(path: str) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    replay_games = 0
    staging_games = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for name in files:
            if name != "replay_buffer.jsonl.gz" and not (
                name.startswith("staging/") and name.endswith(".jsonl.gz")
            ):
                continue
            local = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=name,
                token=token,
                local_dir=temp_dir,
            )
            games = count_lines(local)
            if name == "replay_buffer.jsonl.gz":
                replay_games = games
            else:
                staging_games += games

    summary = {
        "replayGames": replay_games,
        "stagingGames": staging_games,
        "totalGames": replay_games + staging_games,
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output = Path("frontend/public/training-status.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Published training summary: {summary['totalGames']} games")


if __name__ == "__main__":
    main()
