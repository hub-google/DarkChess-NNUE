"""Single HF writer: verify, de-duplicate and store immutable batch from 15 artifacts."""
import glob
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from validate_replay import validate_record

REPO_ID = "hub-google/DarkChess-NNUE-Data"
MAX_ATTEMPTS = 7


def merge_and_upload():
    directory = Path(os.environ.get("OUTPUT_DIR", "output_data"))
    sources = sorted(Path(f) for f in glob.glob(str(directory / "*.jsonl.gz")))
    if not sources:
        raise RuntimeError("No completed games to upload; original artifacts are retained.")
    records = {}
    for source in sources:
        found = 0
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                game = json.loads(line)
                validate_record(game)
                game_id = game["id"]
                serialized = json.dumps(game, sort_keys=True, separators=(",", ":"))
                if game_id in records and records[game_id] != serialized:
                    raise RuntimeError(f"Conflicting replay records for {game_id}")
                records[game_id] = serialized
                found += 1
        if not found:
            raise RuntimeError(f"Empty completed-game artifact: {source}")
    if not records:
        raise RuntimeError("No valid completed games.")
    lines = [records[game_id] for game_id in sorted(records)]
    plain = ("\n".join(lines) + "\n").encode("utf-8")
    digest = hashlib.sha256(plain).hexdigest()[:20]
    earliest = min(int(json.loads(line)["ts"]) // 1000 for line in lines)
    if earliest <= 0:
        raise RuntimeError("Invalid game timestamp.")
    date = time.strftime("%Y%m%d", time.gmtime(earliest))
    filename = f"batch_{earliest}_{digest}.jsonl.gz"
    folder = os.environ.get("PATH_IN_REPO", f"staging/worker_{os.environ.get('WORKER_ID', '16')}/{date}")
    remote_path = f"{folder.rstrip('/')}/{filename}"
    archive = directory / filename
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            gz.write(plain)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; no upload attempted.")
    api = HfApi(token=token)
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if not api.file_exists(repo_id=REPO_ID, repo_type="dataset", filename=remote_path):
                api.upload_file(
                    path_or_fileobj=str(archive), path_in_repo=remote_path,
                    repo_id=REPO_ID, repo_type="dataset",
                    commit_message=f"Verified wave {os.environ.get('WAVE', '?')}: {len(records)} games",
                )
            # Verify real durable remote bytes; retries use the exact same key.
            with tempfile.TemporaryDirectory() as tmp:
                remote = hf_hub_download(
                    repo_id=REPO_ID, repo_type="dataset", filename=remote_path,
                    token=token, local_dir=tmp, force_download=True,
                )
                actual = hashlib.sha256(Path(remote).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Remote SHA-256 differs from local batch: {remote_path}")
            print(f"[HF Writer] VERIFIED {len(records)} valid unique games at {remote_path}")
            break
        except Exception as error:
            last_error = error
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"HF upload unverified after {MAX_ATTEMPTS} attempts; artifacts retained: {last_error}") from error
            delay = min(90, 4 * 2 ** (attempt - 1))
            print(f"[HF Writer] {attempt}/{MAX_ATTEMPTS} failed: {error}; wait {delay}s")
            time.sleep(delay)
    for source in sources:
        if source.resolve() != archive.resolve():
            source.unlink()
    archive.unlink()
    return len(records)


if __name__ == "__main__":
    merge_and_upload()
