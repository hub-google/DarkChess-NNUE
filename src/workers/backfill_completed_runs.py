"""One-shot replay recovery using the exact same verified Worker-16 uploader.

Download the immutable completed-game artifacts of earlier Actions runs, merge
their original .jsonl.gz files, upload via upload_batch.merge_and_upload, then
write a receipt only after HF read-back verifies the remote batch SHA256.
Never manufacture games from partial search checkpoints.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from upload_batch import REPO_ID, merge_and_upload


def gh(*args):
    return subprocess.run(["gh", *map(str, args)], check=True,
                          capture_output=True, text=True).stdout


def backfill(run_ids):
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; refusing to mark any run uploaded")
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN missing; cannot access original artifacts")
    api = HfApi(token=token)
    for run_id in run_ids:
        receipt = f"receipts/backfill/{run_id}.json"
        if api.file_exists(repo_id=REPO_ID, repo_type="dataset", filename=receipt):
            # Receipt is only written after remote batch SHA256 read-back passes.
            print(f"[Backfill] Run {run_id} previously verified; no duplicate upload.", flush=True)
            continue
        response = json.loads(gh("api", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"))
        archives = [
            a for a in response.get("artifacts", [])
            if not a.get("expired") and
            (re.fullmatch(rf"selfplay-worker-\d+-{run_id}", a["name"]) or
             re.fullmatch(rf"selfplay-wave-\d+-worker-\d+-{run_id}", a["name"]))
        ]
        print(f"[Backfill] Run {run_id} artifacts={len(archives)}", flush=True)
        if not archives:
            raise RuntimeError(f"Run {run_id} has no available worker artifacts; nothing was uploaded")
        with tempfile.TemporaryDirectory(prefix=f"backfill-{run_id}-") as tmp:
            root = Path(tmp)
            output = root / "output_data"
            output.mkdir()
            source_count = 0
            for artifact in archives:
                location = root / f"artifact-{artifact['id']}"
                location.mkdir()
                gh("run", "download", run_id, "--repo", repo,
                   "--name", artifact["name"], "--dir", location)
                for original in sorted(location.rglob("*.jsonl.gz")):
                    destination = output / original.name
                    if destination.exists():
                        raise RuntimeError(f"Duplicate replay basename: {original.name}")
                    shutil.copyfile(original, destination)
                    source_count += 1
            print(f"[Backfill] Run {run_id} completed game archives={source_count}", flush=True)
            if not source_count:
                print(f"[Backfill] Run {run_id} contains only checkpoints; no fake games uploaded.", flush=True)
                continue
            old = os.environ.get("OUTPUT_DIR")
            try:
                os.environ["OUTPUT_DIR"] = str(output)
                total_games = merge_and_upload()
            finally:
                if old is None:
                    os.environ.pop("OUTPUT_DIR", None)
                else:
                    os.environ["OUTPUT_DIR"] = old
            # merge_and_upload verifies the HF file's actual bytes before returning.
            receipt_data = json.dumps({
                "run_id": run_id, "games": total_games,
                "input_archives": source_count, "verified_by": "upload_batch.merge_and_upload",
                "verified_at": int(time.time()),
            }, sort_keys=True).encode("utf-8") + b"\n"
            file = root / "receipt.json"
            file.write_bytes(receipt_data)
            for attempt in range(1, 5):
                try:
                    api.upload_file(
                        path_or_fileobj=str(file), path_in_repo=receipt,
                        repo_id=REPO_ID, repo_type="dataset",
                        commit_message=f"Verify original completed self-play run {run_id}")
                    p = hf_hub_download(
                        repo_id=REPO_ID, repo_type="dataset", filename=receipt,
                        token=token, local_dir=str(root / "receipt-readback"),
                        force_download=True)
                    if Path(p).read_bytes() != receipt_data:
                        raise RuntimeError(f"HF receipt SHA mismatch: {receipt}")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(30, 2 ** attempt))
            print(f"[Backfill] VERIFIED run={run_id} games={total_games} HF_receipt={receipt}", flush=True)


if __name__ == "__main__":
    raw = os.environ.get("BACKFILL_RUN_IDS", "").strip()
    if not raw:
        raise SystemExit("BACKFILL_RUN_IDS is required")
    ids = [int(v) for v in raw.replace(",", " ").split()]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit("Invalid or duplicated backfill run IDs")
    backfill(ids)
