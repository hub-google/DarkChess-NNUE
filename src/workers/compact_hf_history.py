"""Reclaim HF dataset history after a verified successful upload.

Squashing removes old Git/LFS revisions, not files in the current branch.
Callers must already have verified a durable upload/read-back.
"""

import os
import time

from huggingface_hub import HfApi

REPO_ID = "hub-google/DarkChess-NNUE-Data"
BRANCH = "main"


def current_files(api):
    """Snapshot current replay, staging batches and all accompanying receipts."""
    files = {}
    for item in api.list_repo_tree(
        repo_id=REPO_ID, repo_type="dataset", revision=BRANCH,
        recursive=True, expand=True,
    ):
        if not hasattr(item, "size"):
            continue  # RepoFolder
        lfs = getattr(item, "lfs", None) or {}
        files[item.path] = (
            item.size,
            getattr(item, "blob_id", None),
            lfs.get("sha256") if isinstance(lfs, dict) else None,
        )
    if not files or not any(
        path == "replay_buffer.jsonl.gz" or path.startswith("staging/")
        for path in files
    ):
        raise RuntimeError("No active replay or staging files; refusing history cleanup")
    return files


def compact_history(api, attempts=3, sleep=time.sleep):
    """Squash history, then verify every live file and the remaining commit."""
    before = current_files(api)
    print(f"[HF history] Protecting {len(before)} live files before squash.", flush=True)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            commits = api.list_repo_commits(
                repo_id=REPO_ID, repo_type="dataset", revision=BRANCH
            )
            # An earlier attempt may have completed even if HTTP timed out.
            if len(commits) > 1:
                api.super_squash_history(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    branch=BRANCH,
                    commit_message="保留目前棋譜，清除已失效的 HF 歷史版本",
                )
            after = current_files(api)
            if after != before:
                missing = sorted(set(before) - set(after))
                changed = sorted(
                    path for path in before.keys() & after.keys()
                    if before[path] != after[path]
                )
                added = sorted(set(after) - set(before))
                raise RuntimeError(
                    f"Live dataset tree changed after history cleanup: "
                    f"missing={missing[:10]}, changed={changed[:10]}, "
                    f"added={added[:10]}"
                )
            remaining = len(api.list_repo_commits(
                repo_id=REPO_ID, repo_type="dataset", revision=BRANCH
            ))
            if remaining != 1:
                raise RuntimeError(
                    f"Expected 1 HF commit after squash; found {remaining}"
                )
            print(
                f"[HF history] VERIFIED: {len(after)} files intact, "
                "1 commit retained.", flush=True
            )
            print(
                "[HF history] Storage quota may take up to 36 hours to update.",
                flush=True
            )
            return
        except Exception as exc:
            last_error = exc
            print(
                f"[HF history] Attempt {attempt}/{attempts} failed: {exc}",
                flush=True
            )
            if attempt < attempts:
                sleep(min(15 * attempt, 30))
    raise RuntimeError(
        f"HF history cleanup failed; stop workflow, preserve upload: {last_error}"
    )


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; refusing cleanup")
    compact_history(HfApi(token=token))


if __name__ == "__main__":
    main()
