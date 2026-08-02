"""Verify every historical shard manifest, then atomically clear staging."""
import json
import os

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "hub-google/DarkChess-NNUE-Data"
SHARDS = ["fresh"] + [f"worker_{index}" for index in range(1, 21)]


def finalize():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    archive_files = {
        entry.path for entry in api.list_repo_tree(
            repo_id=REPO_ID,
            path_in_repo="archive",
            recursive=True,
            repo_type="dataset",
        )
    }
    manifests = []
    for shard in SHARDS:
        archive_name = f"archive/{shard}.jsonl.gz"
        manifest_name = f"archive/{shard}.manifest.json"
        if archive_name not in archive_files or manifest_name not in archive_files:
            raise RuntimeError(f"archive is incomplete: missing {shard}")
        manifest_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=manifest_name,
            repo_type="dataset",
            token=token,
        )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("shard") != shard:
            raise RuntimeError(f"manifest shard mismatch: {shard}")
        if not manifest.get("sha256") or manifest.get("archive_bytes", 0) <= 0:
            raise RuntimeError(f"invalid archive manifest: {shard}")
        manifests.append(manifest)

    source_files = sum(item["source_file_count"] for item in manifests)
    records = sum(item["record_count"] for item in manifests)
    print(f"Verified 21 archive shards: {source_files} files, {records} records")
    if source_files < 500_000 or records <= 0:
        raise RuntimeError("archive totals are unexpectedly small; refusing deletion")

    api.delete_folder(
        path_in_repo="staging",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Clear staging after archiving {source_files} source files",
    )
    try:
        remaining = [
            entry.path for entry in api.list_repo_tree(
                repo_id=REPO_ID,
                path_in_repo="staging",
                recursive=True,
                repo_type="dataset",
            )
        ]
    except Exception as error:
        if "404" in str(error) or "not found" in str(error).lower():
            remaining = []
        else:
            raise
    if remaining:
        raise RuntimeError(f"staging deletion incomplete: {len(remaining)} entries remain")
    print("Staging successfully archived and cleared to zero.")


if __name__ == "__main__":
    finalize()
