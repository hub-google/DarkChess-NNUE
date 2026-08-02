"""
Consolidate Replay Buffer

Downloads all staging .jsonl.gz files from the HF dataset repo,
merges them into a single replay_buffer.jsonl.gz (capped at 500k games),
uploads the buffer, and deletes the old staging files.

Lists only the staging subtree and downloads files one-by-one.  The dataset
also contains a large legacy root archive, so listing the whole repository can
take longer than a GitHub-hosted runner's six-hour limit.
"""
import os
import gzip
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete

MAX_REPLAY_GAMES = 500_000  # Sliding window capacity
REPO_ID = "hub-google/DarkChess-NNUE-Data"
BUFFER_FILE = "replay_buffer.jsonl.gz"
CHUNK_SIZE = 500  # Max delete operations per commit


def list_recent_staging_files(api):
    """List bounded, date-partitioned v2 staging data without walking legacy data."""
    lookback_days = int(os.environ.get("STAGING_LOOKBACK_DAYS", "7"))
    worker_count = int(os.environ.get("SELF_PLAY_WORKER_COUNT", "15"))
    today = datetime.now(timezone.utc).date()
    paths = [
        f"staging/worker_{worker_id}/{(today - timedelta(days=offset)):%Y%m%d}"
        for worker_id in range(1, worker_count + 1)
        for offset in range(lookback_days)
    ]
    # `fresh` is retained for compatibility with the previous uploader.
    paths.append("staging/fresh")

    def list_path(path):
        try:
            return [
                entry.path
                for entry in api.list_repo_tree(
                    repo_id=REPO_ID,
                    path_in_repo=path,
                    recursive=True,
                    repo_type="dataset",
                )
                if getattr(entry, "path", "").endswith(".jsonl.gz")
            ]
        except Exception as error:
            # Missing date partitions are normal when a worker produced no game.
            if "404" not in str(error) and "not found" not in str(error).lower():
                print(f"   Warning: could not list {path}: {error}")
            return []

    files = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(list_path, path) for path in paths]
        for future in as_completed(futures):
            files.extend(future.result())
    files = sorted(set(files), reverse=True)
    max_files = int(os.environ.get("MAX_STAGING_FILES", "2000"))
    if len(files) > max_files:
        print(f"   Limiting {len(files)} staging files to the newest {max_files}.")
        files = files[:max_files]
    return files


def consolidate():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=hf_token)
    temp_dir = "temp_consolidate"
    output_dir = "output_buffer"
    output_path = os.path.join(output_dir, BUFFER_FILE)

    # Clean up any leftover temp dirs
    for d in [temp_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. List only staging.  Never call list_repo_files here: the repository
    # has a large legacy root archive which is unrelated to nightly training.
    print("1. Listing staging files in the dataset repo...")
    try:
        staging_files = list_recent_staging_files(api)
    except Exception as e:
        raise RuntimeError(f"Error listing staging files: {e}") from e

    print(f"   Staging .jsonl.gz files: {len(staging_files)}")

    # 2. Download and read existing replay buffer (if any)
    all_games = []
    successful_staging_files = []
    print("2. Downloading existing replay buffer (if present)...")
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID, filename=BUFFER_FILE,
            repo_type="dataset", token=hf_token, local_dir=temp_dir
        )
        with gzip.open(local_path, 'rt', encoding='utf-8') as gz:
            for line in gz:
                line_str = line.strip()
                if line_str:
                    all_games.append(line_str)
        print(f"   Read {len(all_games)} games from existing buffer.")
    except Exception as e:
        # The first successful consolidation legitimately has no buffer yet.
        print(f"   No readable existing buffer; continuing with staging: {e}")

    # 3. Download staging files one by one
    if staging_files:
        print(f"3. Downloading {len(staging_files)} staging files...")
        def download_staging(sf):
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID, filename=sf,
                    repo_type="dataset", token=hf_token, local_dir=temp_dir
                )
                records = []
                with gzip.open(local_path, 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        line_str = line.strip()
                        if line_str:
                            records.append(line_str)
                return sf, records, None
            except Exception as e:
                return sf, [], e

        downloaded = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_staging, sf) for sf in staging_files]
            for i, future in enumerate(as_completed(futures), start=1):
                sf, records, error = future.result()
                if error is None:
                    all_games.extend(records)
                    downloaded += 1
                    successful_staging_files.append(sf)
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"   Warning: skipping {sf}: {error}")
                if i % 100 == 0:
                    print(f"   Progress: {i}/{len(staging_files)} files processed, accumulated {len(all_games)} games so far...")

        print(f"   Downloaded {downloaded} files, skipped {failed} files.")
    else:
        print("3. No staging files to process.")

    print(f"   Total game records collected: {len(all_games)}")

    if not all_games:
        raise RuntimeError("No games found; refusing to replace the replay buffer.")

    # A run can upload the new replay buffer and then fail before deleting
    # staging files. De-duplicate by immutable game id before applying the
    # sliding window so the retry does not overweight those games.
    deduplicated = {}
    invalid_records = 0
    for game_str in all_games:
        try:
            game = json.loads(game_str)
            game_id = game["id"]
            if not isinstance(game_id, str) or not game_id:
                raise ValueError("missing game id")
            deduplicated[game_id] = game_str
        except Exception:
            invalid_records += 1
    all_games = list(deduplicated.values())
    all_games.sort(key=lambda line: int(json.loads(line).get("ts", 0)))
    print(
        f"   De-duplicated replay data: {len(all_games)} unique games, "
        f"{invalid_records} invalid records skipped."
    )
    if not all_games:
        raise RuntimeError("No valid games remain after replay validation.")

    # 4. Apply sliding window cap
    if len(all_games) > MAX_REPLAY_GAMES:
        print(f"4. Trimming from {len(all_games)} to latest {MAX_REPLAY_GAMES} games.")
        all_games = all_games[-MAX_REPLAY_GAMES:]
    else:
        print(f"4. Game count ({len(all_games)}) within cap, keeping all.")

    # Train on the most recent policy first when a nightly sample cap is used.
    all_games.reverse()

    # 5. Write consolidated replay buffer
    print(f"5. Writing consolidated buffer ({len(all_games)} games)...")
    with gzip.open(output_path, 'wt', encoding='utf-8') as gz:
        for game_str in all_games:
            gz.write(game_str + '\n')

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   Buffer file: {output_path} ({file_size_mb:.1f} MB)")

    # 6. Upload the consolidated buffer
    print("6. Uploading consolidated replay buffer...")
    for attempt in range(1, 6):
        try:
            api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=BUFFER_FILE,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Update consolidated replay buffer ({len(all_games)} games)"
            )
            print(f"   Uploaded {BUFFER_FILE} successfully.")
            break
        except Exception as e:
            if attempt < 5:
                wait = 30 * attempt
                print(f"   Upload error (attempt {attempt}/5): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Upload failed after 5 attempts: {e}") from e

    # 7. Delete staging files from HF repo
    if successful_staging_files:
        print(f"7. Cleaning up {len(successful_staging_files)} downloaded staging files from HF...")
        chunks = [
            successful_staging_files[i:i + CHUNK_SIZE]
            for i in range(0, len(successful_staging_files), CHUNK_SIZE)
        ]

        for i, chunk in enumerate(chunks):
            delete_ops = [CommitOperationDelete(path_in_repo=f) for f in chunk]
            for attempt in range(1, 6):
                try:
                    api.create_commit(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        operations=delete_ops,
                        commit_message=f"Clean staging files (batch {i+1}/{len(chunks)})"
                    )
                    print(f"   Deleted batch {i+1}/{len(chunks)} ({len(chunk)} files)")
                    break
                except Exception as e:
                    if '429' in str(e) or 'rate' in str(e).lower():
                        wait = 60 * attempt
                        print(f"   Rate limited batch {i+1}. Waiting {wait}s...")
                        time.sleep(wait)
                    elif attempt < 5:
                        print(f"   Error batch {i+1} (attempt {attempt}/5): {e}")
                        time.sleep(15 * attempt)
                    else:
                        print(f"   FAILED batch {i+1}: {e}")

            if i < len(chunks) - 1:
                time.sleep(3)

        print("   Staging cleanup complete.")
    else:
        print("7. No staging files to clean up.")

    # 8. Cleanup local temp
    for d in [temp_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print("Done! Replay buffer consolidated successfully.")


if __name__ == '__main__':
    consolidate()
