"""Merge one Hugging Face staging shard into a verified historical archive."""
import gzip
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


REPO_ID = "hub-google/DarkChess-NNUE-Data"


def download_with_retry(filename, token, local_dir):
    error = None
    for attempt in range(1, 6):
        try:
            return hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=token,
                local_dir=local_dir,
            )
        except Exception as exc:
            error = exc
            if attempt < 5:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"failed to download {filename}: {error}")


def archive_shard():
    token = os.environ.get("HF_TOKEN")
    shard = os.environ.get("ARCHIVE_SHARD")
    if not token or not shard:
        raise RuntimeError("HF_TOKEN and ARCHIVE_SHARD are required")
    if shard != "fresh" and not shard.startswith("worker_"):
        raise ValueError(f"invalid archive shard: {shard}")

    api = HfApi(token=token)
    source_path = f"staging/{shard}"
    entries = api.list_repo_tree(
        repo_id=REPO_ID,
        path_in_repo=source_path,
        recursive=True,
        repo_type="dataset",
    )
    source_files = sorted(
        entry.path for entry in entries
        if getattr(entry, "path", "").endswith(".jsonl.gz")
    )
    if not source_files:
        print(f"{source_path} is empty; writing an empty verified shard.")

    work_dir = Path(f"archive_work_{shard}")
    download_dir = work_dir / "downloads"
    output_dir = work_dir / "output"
    shutil.rmtree(work_dir, ignore_errors=True)
    download_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    archive_path = output_dir / f"{shard}.jsonl.gz"
    downloaded = {}
    failures = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(download_with_retry, name, token, str(download_dir)): name
            for name in source_files
        }
        for index, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                downloaded[name] = future.result()
            except Exception as exc:
                failures.append(str(exc))
            if index % 500 == 0:
                print(f"Downloaded {index}/{len(source_files)} files")
    if failures:
        raise RuntimeError(
            f"{len(failures)} source files failed; refusing to publish shard. "
            + "; ".join(failures[:3])
        )

    record_count = 0
    invalid_gzip = []
    with gzip.open(archive_path, "wt", encoding="utf-8") as output:
        for name in source_files:
            try:
                with gzip.open(downloaded[name], "rt", encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            output.write(line.rstrip("\n") + "\n")
                            record_count += 1
            except Exception as exc:
                invalid_gzip.append(f"{name}: {exc}")
    if invalid_gzip:
        raise RuntimeError(
            f"{len(invalid_gzip)} corrupt sources; refusing to publish shard. "
            + "; ".join(invalid_gzip[:3])
        )

    digest = hashlib.sha256()
    with open(archive_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "format": 1,
        "shard": shard,
        "source_path": source_path,
        "source_file_count": len(source_files),
        "record_count": record_count,
        "archive_bytes": archive_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    manifest_path = output_dir / f"{shard}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for attempt in range(1, 6):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"archive/{archive_path.name}",
                        path_or_fileobj=str(archive_path),
                    ),
                    CommitOperationAdd(
                        path_in_repo=f"archive/{manifest_path.name}",
                        path_or_fileobj=str(manifest_path),
                    ),
                ],
                commit_message=f"Archive historical staging shard {shard}",
            )
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(15 * attempt)
    print(json.dumps(manifest))


if __name__ == "__main__":
    archive_shard()
