"""Recover durable failed self-play artifacts without re-uploading acknowledged games.

Runs periodically from the consolidated self-play maintenance job. GitHub artifact archives are
immutable for 14 days. An HF receipt is written only after the replay has been
uploaded AND independently verified by upload_batch.merge_and_upload().
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from upload_batch import REPO_ID, merge_and_upload


def gh_json(resource):
    process = subprocess.run(
        ["gh", "api", resource],
        check=True, capture_output=True, text=True,
    )
    return json.loads(process.stdout)


def recover():
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing: cannot recover staged self-play")
    if not os.environ.get("GH_TOKEN"):
        raise RuntimeError("GH_TOKEN missing: cannot download Actions artifacts")
    api = HfApi(token=token)
    cutoff = time.time() - 13 * 86400
    recovered = 0
    max_groups = max(1, int(os.environ.get("MAX_RECOVERY_GROUPS", "8")))

    # Examine all failures within artifact retention, not just the last 30
    # hourly watchdog runs. Success runs have no failed HF-writer data.
    from datetime import datetime
    for page in range(1, 11):
        runs = gh_json(
            f"repos/{repo}/actions/workflows/self_play.yml/runs?per_page=100&page={page}"
        ).get("workflow_runs", [])
        for run in runs:
            timestamp = run.get("created_at", "").replace("Z", "+00:00")
            if datetime.fromisoformat(timestamp).timestamp() < cutoff:
                print(f"[Recovery] Reached 13-day artifact retention cutoff.")
                return
            # The overall run may succeed after a failed earlier uploader.
            # Inspect *all* completed rounds and recover only unverified waves.
            if run.get("status") != "completed":
                continue
            run_id = int(run["id"])
            # Never rely on mutable branch HEAD for a recovery run's input.
            artifact_list = gh_json(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100")
            artifacts = [a for a in artifact_list.get("artifacts", []) if not a.get("expired")]
            groups = defaultdict(list)
            sources_by_wave = defaultdict(list)
            for a in artifacts:
                name = a["name"]
                retry = re.fullmatch(r"hf-upload-recovery-wave-(\d+)-(\d+)", name)
                worker = re.fullmatch(r"selfplay-wave-(\d+)-worker-(\d+)-(\d+)", name)
                single_retry = re.fullmatch(r"hf-upload-recovery-run-(\d+)", name)
                single_worker = re.fullmatch(r"selfplay-worker-(\d+)-(\d+)", name)
                legacy = re.fullmatch(r"self-play-recovery-\d+-\d+", name)
                if retry and int(retry.group(2)) == run_id:
                    groups[f"wave-{retry.group(1)}"] = [a]
                elif worker and int(worker.group(3)) == run_id:
                    sources_by_wave[worker.group(1)].append(a)
                elif single_retry and int(single_retry.group(1)) == run_id:
                    groups["run"] = [a]
                elif single_worker and int(single_worker.group(2)) == run_id:
                    sources_by_wave["run"].append(a)
                elif legacy:
                    groups[f"legacy-{a['id']}"] = [a]
            # A worker may finish but the writer may be cancelled before creating
            # its recovery artifact. Consult the completed writer job's result.
            if sources_by_wave:
                jobs = gh_json(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100").get("jobs", [])
                verified_waves = set()
                for job in jobs:
                    match = re.search(
                        r"Wave\s+(\d+)\s*/\s*worker 16 HF writer",
                        job.get("name", ""), re.I,
                    )
                    if match and job.get("conclusion") == "success":
                        verified_waves.add(match.group(1))
                    if "第16號棋譜上傳員" in job.get("name", "") and job.get("conclusion") == "success":
                        verified_waves.add("run")
                # Cancelled runs might never schedule the writer.
                for wave, worker_artifacts in sources_by_wave.items():
                    if wave not in verified_waves:
                        groups.setdefault("run" if wave == "run" else f"wave-{wave}", worker_artifacts)

            for key, parts in sorted(groups.items()):
                if not parts:
                    continue
                receipt = f"receipts/recovery/{run_id}/{key}.json"
                if api.file_exists(repo_id=REPO_ID, repo_type="dataset", filename=receipt):
                    continue
                with tempfile.TemporaryDirectory(prefix="hf-recovery-") as temp:
                    root = Path(temp)
                    output = root / "output_data"
                    output.mkdir()
                    for a in parts:
                        extracted = root / f"artifact-{a['id']}"
                        extracted.mkdir()
                        subprocess.run(
                            ["gh", "run", "download", str(run_id), "--repo", repo,
                             "--name", a["name"], "--dir", str(extracted)],
                            check=True,
                        )
                        for source in extracted.rglob("*.jsonl.gz"):
                            destination = output / source.name
                            if destination.exists():
                                raise RuntimeError(f"Duplicate recovery file name: {source.name}")
                            shutil.copy2(source, destination)
                    if not list(output.glob("*.jsonl.gz")):
                        raise RuntimeError(f"No recoverable games in {run_id}/{key}")
                    old = os.environ.get("OUTPUT_DIR")
                    try:
                        os.environ["OUTPUT_DIR"] = str(output)
                        count = merge_and_upload()
                    finally:
                        if old is None:
                            os.environ.pop("OUTPUT_DIR", None)
                        else:
                            os.environ["OUTPUT_DIR"] = old
                    receipt_file = root / "receipt.json"
                    receipt_file.write_text(
                        json.dumps({"run": run_id, "group": key, "games": count}) + "\n",
                        encoding="utf-8",
                    )
                    api.upload_file(
                        path_or_fileobj=str(receipt_file), path_in_repo=receipt,
                        repo_id=REPO_ID, repo_type="dataset",
                        commit_message=f"Acknowledge recovered self-play {run_id}/{key}",
                    )
                    verified = hf_hub_download(
                        repo_id=REPO_ID, repo_type="dataset", filename=receipt,
                        token=token, local_dir=str(root / "receipt-check"),
                        force_download=True,
                    )
                    if Path(verified).read_bytes() != receipt_file.read_bytes():
                        raise RuntimeError(f"Recovery receipt verification failed: {receipt}")
                    print(f"[Recovery] VERIFIED {count} games from run {run_id}/{key}")
                    recovered += 1
                    if recovered >= max_groups:
                        print(f"[Recovery] Bounded to {max_groups} groups this cycle.")
                        return
            if recovered >= max_groups:
                return
        if len(runs) < 100:
            break
    else:
        print("[Recovery] Reached 1000 recent runs; next pass will retry.")

    print(f"[Recovery] Recovered {recovered} groups; all others acknowledged or none available.")


if __name__ == "__main__":
    recover()
