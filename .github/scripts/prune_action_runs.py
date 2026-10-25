"""Keep one latest GitHub Actions run per *logical* task, without losing HF data.

The three workflow files are shared by several logical jobs. Categorize runs by
the event and stable run-name, not merely by workflow file. Never cancel any
running run to tidy history. A completed self-play round can be deleted only if
all its available game artifacts have a verified HF writer or recovery receipt.

Run from existing jobs (no extra workflow):
  python .github/scripts/prune_action_runs.py --scope all
  python .github/scripts/prune_action_runs.py --scope self_play

GH_TOKEN needs actions:write. HF_TOKEN is optional: without it, unsafe
self-play runs are kept until the next scheduled authenticated maintenance.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "hub-google/DarkChess-NNUE")
WORKFLOWS = {
    "self_play": "self_play.yml",
    "train": "train.yml",
    "deploy": "deploy_pages.yml",
}
HF_DATASET = "hub-google/DarkChess-NNUE-Data"

# New run-name is stable even when a push commit message changes.
WATCHDOG_TITLE = "每小時恢復檢查"
MAINTENANCE_TITLE = "棋譜修復與統計"
KEEPALIVE_TITLE = "每月排程維護"


def gh_api(resource, method="GET"):
    cmd = ["gh", "api"]
    if method != "GET":
        cmd.extend(["--method", method])
    cmd.append(f"repos/{REPOSITORY}/{resource}")
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, timeout=50,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def pages(resource, key, max_pages=30):
    """Fetch actual pages, not just GitHub's default first 30 entries."""
    results = []
    for page in range(1, max_pages + 1):
        payload = gh_api(f"{resource}{'&' if '?' in resource else '?'}per_page=100&page={page}")
        records = payload.get(key, [])
        results.extend(records)
        if len(records) < 100:
            break
    else:
        raise RuntimeError(f"pagination limit reached: {resource}")
    return results


def run_group(workflow, run):
    event = run.get("event")
    title = run.get("display_title") or ""
    if workflow == "self_play":
        if event == "workflow_dispatch":
            return "self-play"
        if event == "schedule":
            if title.startswith(WATCHDOG_TITLE):
                return "watchdog"
            if title.startswith(MAINTENANCE_TITLE):
                return "HF maintenance"
            if title.startswith(KEEPALIVE_TITLE):
                return "keepalive"
            return "legacy scheduled check"
        return "legacy self-play setup"
    if workflow == "train":
        if event == "push" and not (
            title.startswith("模型訓練與升級") or "[train-now]" in title
        ):
            return "python regression"
        return "NNUE training"
    return "GitHub Pages"


def _writer_waves(run_id):
    jobs = pages(f"actions/runs/{run_id}/jobs?filter=latest", "jobs")
    verified = set()
    for job in jobs:
        if job.get("conclusion") != "success":
            continue
        # Both the old reusable-wave layout and the new consolidated layout.
        match = re.search(r"(?:Wave\s+(\d+)\s*/\s*worker 16 HF writer|第(\d+)輪[／/]第16號棋譜上傳員)", job.get("name", ""), re.I)
        if match:
            verified.add(match.group(1) or match.group(2))
    return verified


def self_play_artifacts_safe(run_id, hf_api=None):
    """A deletion must not destroy the last recoverable copy of self-play."""
    artifacts = pages(f"actions/runs/{run_id}/artifacts", "artifacts")
    valid = [a for a in artifacts if not a.get("expired")]
    workers = set()
    explicit_recovery = set()
    for artifact in valid:
        name = artifact.get("name", "")
        worker = re.fullmatch(rf"selfplay-wave-(\d+)-worker-\d+-{run_id}", name)
        rescue = re.fullmatch(rf"hf-upload-recovery-wave-(\d+)-{run_id}", name)
        legacy = re.fullmatch(r"self-play-recovery-\d+-\d+", name)
        if worker:
            workers.add(worker.group(1))
        if rescue:
            explicit_recovery.add(f"wave-{rescue.group(1)}")
        if legacy:
            explicit_recovery.add(f"legacy-{artifact['id']}")
    if not workers and not explicit_recovery:
        return True
    verified = _writer_waves(run_id) if workers else set()
    unresolved = {f"wave-{w}" for w in workers if w not in verified} | explicit_recovery
    if not unresolved:
        return True
    if hf_api is None:
        print(f"[Prune] Keep #{run_id}: pending HF recovery {sorted(unresolved)}, no authenticated verifier.")
        return False
    for group in sorted(unresolved):
        path = f"receipts/recovery/{run_id}/{group}.json"
        if not hf_api.file_exists(repo_id=HF_DATASET, repo_type="dataset", filename=path):
            print(f"[Prune] Keep #{run_id}: missing verified HF recovery receipt {path}.")
            return False
    return True


def choose_deletions(workflow, runs, protected_run_id=None):
    """Select only *completed* older records from each logical task group."""
    groups = defaultdict(list)
    for run in runs:
        groups[run_group(workflow, run)].append(run)
    selected = []
    for group, records in groups.items():
        # Legacy scheduled/push records have been replaced by labelled tasks.
        # The new run-name is now the durable record for those logical jobs.
        # Old scheduled/push-only orchestration runs will never recur under the
        # new task names. They can all disappear; manual rounds remain protected.
        keep_count = 0 if workflow == "self_play" and group in (
            "legacy scheduled check", "legacy self-play setup"
        ) else 1
        records.sort(key=lambda r: (r.get("created_at", ""), int(r["id"])), reverse=True)
        for old in records[keep_count:]:
            if int(old["id"]) == int(protected_run_id or 0):
                continue
            if old.get("status") == "completed":
                selected.append((group, old))
    return selected


def prune(scope="all", dry_run=False):
    if not os.getenv("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN required for Actions history cleanup")
    hf_api = None
    if os.getenv("HF_TOKEN"):
        from huggingface_hub import HfApi
        hf_api = HfApi(token=os.environ["HF_TOKEN"])
    protected = int(os.environ.get("GITHUB_RUN_ID") or 0)
    deleted = 0
    retained = 0
    inspected = 0
    for workflow, filename in WORKFLOWS.items():
        if scope not in ("all", workflow):
            continue
        runs = pages(f"actions/workflows/{filename}/runs", "workflow_runs")
        inspected += len(runs)
        for group, candidate in choose_deletions(workflow, runs, protected):
            run_id = int(candidate["id"])
            try:
                # Refresh status immediately before acting on mutable history.
                current = gh_api(f"actions/runs/{run_id}")
                if current.get("status") != "completed":
                    retained += 1
                    continue
                if workflow == "self_play" and group == "self-play":
                    if not self_play_artifacts_safe(run_id, hf_api):
                        retained += 1
                        continue
                if dry_run:
                    print(f"[Prune] DRY RUN would remove {workflow}/{group} #{run_id}")
                    continue
                gh_api(f"actions/runs/{run_id}", method="DELETE")
                print(f"[Prune] Deleted older completed {workflow}/{group} #{run_id}.")
                deleted += 1
            except Exception as error:
                # A failed HF check or GitHub API must never lose the game data.
                retained += 1
                print(f"[Prune] Preserved #{run_id} after error: {error}", file=sys.stderr)
    print(f"[Prune] inspected={inspected} deleted={deleted} preserved={retained} scope={scope}")
    return deleted, retained


def main():
    parser = argparse.ArgumentParser(description="Keep the latest run per logical task.")
    parser.add_argument("--scope", choices=("all", *WORKFLOWS), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prune(scope=args.scope, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
