"""Archive HF staging with one Git tree fetch and Git-Xet sparse checkouts.

This deliberately avoids ``list_repo_tree`` and per-file ``hf_hub_download``.
Git transfers the repository tree as a pack, while Git-Xet reconstructs only the
currently selected staging shard from CAS.  At most one source shard and one
archive are present on disk at a time.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


REPO_ID = "hub-google/DarkChess-NNUE-Data"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"
SHARDS = ["fresh"] + [f"worker_{index}" for index in range(1, 21)]


def run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_shard(repo_dir: Path, shard: str, output_dir: Path) -> tuple[Path, dict]:
    source_root = repo_dir / "staging" / shard
    source_files = sorted(source_root.rglob("*.jsonl.gz")) if source_root.exists() else []
    archive_path = output_dir / f"{shard}.jsonl.gz"
    record_count = 0

    with gzip.open(archive_path, "wt", encoding="utf-8") as output:
        for index, source_path in enumerate(source_files, start=1):
            with gzip.open(source_path, "rt", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        output.write(line.rstrip("\n") + "\n")
                        record_count += 1
            if index % 1000 == 0:
                print(f"Merged {index}/{len(source_files)} files for {shard}", flush=True)

    manifest = {
        "format": 2,
        "shard": shard,
        "source_path": f"staging/{shard}",
        "source_file_count": len(source_files),
        "record_count": record_count,
        "archive_bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
    }
    return archive_path, manifest


def upload_archive(api: HfApi, archive_path: Path, manifest: dict, output_dir: Path) -> None:
    shard = manifest["shard"]
    manifest_path = output_dir / f"{shard}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def completed_manifest(repo_dir: Path, shard: str, staging_tree: str) -> dict | None:
    manifest_path = repo_dir / "archive" / f"{shard}.manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("format") == 2
        and manifest.get("shard") == shard
        and manifest.get("source_staging_tree") == staging_tree
        and manifest.get("sha256")
    ):
        return manifest
    return None


def archive_repository() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    work_dir = Path("archive_work")
    repo_dir = work_dir / "repo"
    output_dir = work_dir / "output"
    shutil.rmtree(work_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    # Authentication is installed into Git's credential store by the workflow.
    # Blob filtering keeps Xet payloads out until a sparse path is checked out.
    run_git("clone", "--filter=blob:none", "--no-checkout", "--depth=1", REPO_URL, str(repo_dir))
    run_git("sparse-checkout", "init", "--no-cone", cwd=repo_dir)
    source_revision = run_git("rev-parse", "HEAD", cwd=repo_dir)
    source_staging_tree = run_git("rev-parse", f"{source_revision}:staging", cwd=repo_dir)
    print(f"Archiving immutable staging tree {source_staging_tree} at {source_revision}", flush=True)

    api = HfApi(token=token)
    manifests = []
    for shard in SHARDS:
        sparse_path = f"/staging/{shard}/"
        run_git(
            "sparse-checkout",
            "set",
            "--no-cone",
            sparse_path,
            "/archive/*.manifest.json",
            cwd=repo_dir,
        )
        run_git("checkout", "--force", source_revision, cwd=repo_dir)
        existing = completed_manifest(repo_dir, shard, source_staging_tree)
        if existing:
            manifests.append(existing)
            print(f"Reusing verified archive for {shard} from the same staging tree", flush=True)
            continue
        archive_path, manifest = merge_shard(repo_dir, shard, output_dir)
        manifest["source_revision"] = source_revision
        manifest["source_staging_tree"] = source_staging_tree
        upload_archive(api, archive_path, manifest, output_dir)
        manifests.append(manifest)
        archive_path.unlink()
        (output_dir / f"{shard}.manifest.json").unlink()
        print(json.dumps(manifest), flush=True)

    source_files = sum(item["source_file_count"] for item in manifests)
    records = sum(item["record_count"] for item in manifests)
    if source_files < 500_000 or records <= 0:
        raise RuntimeError(
            f"archive totals are unexpectedly small ({source_files} files, {records} records); "
            "refusing deletion"
        )

    # The workflow shares a concurrency lock with self-play, but also verify the
    # remote tree directly so uploads from outside Actions can never be erased.
    run_git("fetch", "--depth=1", "origin", "main", cwd=repo_dir)
    current_staging_tree = run_git("rev-parse", "FETCH_HEAD:staging", cwd=repo_dir)
    if current_staging_tree != source_staging_tree:
        raise RuntimeError(
            "staging changed during archival; archives are safe but staging will not be deleted"
        )

    api.delete_folder(
        path_in_repo="staging",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Clear verified staging tree {source_staging_tree} ({source_files} files)",
    )
    print(f"Archived {source_files} files / {records} records and cleared staging.", flush=True)


if __name__ == "__main__":
    archive_repository()
