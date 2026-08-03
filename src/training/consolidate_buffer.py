"""
Consolidate Replay Buffer

Downloads a cutoff-bounded snapshot of staging files from the HF dataset repo,
merges them into a single replay_buffer.jsonl.gz (capped at 500k games),
uploads and verifies the buffer, then deletes only the files in that snapshot.

Lists only the staging subtree and downloads files one-by-one.  The dataset
also contains a large legacy root archive, so listing the whole repository can
take longer than a GitHub-hosted runner's six-hour limit.
"""
import os
import gzip
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

MAX_REPLAY_GAMES = 500_000  # Sliding window capacity
MAX_REPLAY_BYTES = int(os.environ.get("MAX_REPLAY_BUFFER_BYTES", str(2 * 1024**3)))
REPO_ID = "hub-google/DarkChess-NNUE-Data"
BUFFER_FILE = "replay_buffer.jsonl.gz"
DELETE_BATCH_SIZE = 500


def staging_timestamp(path):
    match = re.search(r"(?:batch|data)_(\d+)", path)
    if not match:
        return 0
    timestamp = int(match.group(1))
    # Historical data_* names used milliseconds while the current batch_*
    # uploader uses seconds. Normalize before comparing with the cutoff.
    return timestamp // 1000 if timestamp >= 100_000_000_000 else timestamp


def list_staging_files(api):
    """List all worker staging subtrees without walking the legacy repo root."""
    # Historical workflows used workers 16-20, while the current workflow uses
    # 1-15. Keep all of them visible until compaction cleanup is complete.
    worker_count = int(os.environ.get("STAGING_WORKER_COUNT", "20"))
    paths = [
        f"staging/worker_{worker_id}"
        for worker_id in range(1, worker_count + 1)
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
    return sorted(set(files))


def validate_buffer(path, expected_count):
    """Verify that the committed replay buffer is readable and de-duplicated."""
    game_ids = set()
    previous_timestamp = None
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            game = json.loads(line)
            game_id = game["id"]
            timestamp = int(game.get("ts", 0))
            if not isinstance(game_id, str) or not game_id:
                raise RuntimeError("replay buffer contains a missing game id")
            if game_id in game_ids:
                raise RuntimeError(f"replay buffer contains duplicate game id {game_id}")
            # The file is newest-first so training sees the latest policy first.
            if previous_timestamp is not None and timestamp > previous_timestamp:
                raise RuntimeError("replay buffer is not ordered newest-first")
            game_ids.add(game_id)
            previous_timestamp = timestamp
            count += 1
    if count != expected_count:
        raise RuntimeError(
            f"replay buffer verification found {count} games; expected {expected_count}"
        )
    if count > MAX_REPLAY_GAMES:
        raise RuntimeError(f"replay buffer exceeds the {MAX_REPLAY_GAMES}-game cap")
    if os.path.getsize(path) > MAX_REPLAY_BYTES:
        raise RuntimeError(
            f"replay buffer exceeds the {MAX_REPLAY_BYTES}-byte compressed-size cap"
        )
    return count


def write_size_bounded_buffer(path, newest_first_games):
    """Write newest games and discard oldest games until both caps are met."""
    retained = list(newest_first_games[:MAX_REPLAY_GAMES])
    while retained:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for game_str in retained:
                handle.write(game_str + "\n")
        size = os.path.getsize(path)
        if size <= MAX_REPLAY_BYTES:
            return retained, size
        # Compressed size is close to proportional for a large JSONL stream.
        # Keep a safety margin, then repeat to enforce the exact hard limit.
        keep = max(1, int(len(retained) * MAX_REPLAY_BYTES / size * 0.98))
        if keep >= len(retained):
            keep = len(retained) - 1
        retained = retained[:keep]
    raise RuntimeError("No replay game fits within the compressed-size cap")


def select_snapshot_files(paths, cutoff_epoch):
    """Return only complete batches whose upload timestamp is in the snapshot."""
    return sorted(
        path for path in paths
        if staging_timestamp(path) <= cutoff_epoch
    )


def delete_snapshot_files_and_verify(api, snapshot_files):
    """Delete only committed snapshot files, preserving concurrent uploads."""
    remaining = set(snapshot_files)
    if not remaining:
        print("   Snapshot contains no staging files to delete.")
        return

    last_error = None
    for attempt in range(1, 6):
        # A previous commit may have succeeded even if its HTTP response failed.
        remaining.intersection_update(list_staging_files(api))
        if not remaining:
            print("   Deleted and verified all snapshot staging files.")
            return

        paths = sorted(remaining)
        try:
            for start in range(0, len(paths), DELETE_BATCH_SIZE):
                chunk = paths[start:start + DELETE_BATCH_SIZE]
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=[
                        CommitOperationDelete(path_in_repo=path)
                        for path in chunk
                    ],
                    commit_message=(
                        f"Delete {len(chunk)} replay-buffer snapshot batches"
                    ),
                )
        except Exception as error:
            # Self-play may commit a different path at the same time. Refresh
            # repository state and retry only snapshot paths still present.
            last_error = error

        remaining.intersection_update(list_staging_files(api))
        if not remaining:
            print("   Deleted and verified all snapshot staging files.")
            return
        if attempt < 5:
            wait = 15 * attempt
            print(
                f"   {len(remaining)} snapshot files remain after cleanup "
                f"attempt {attempt}/5; retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"failed to delete {len(remaining)} snapshot files after 5 attempts: "
        f"{last_error}"
    )


def consolidate():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=hf_token)
    cutoff_epoch = int(os.environ.get("CONSOLIDATION_CUTOFF_EPOCH", time.time()))
    temp_dir = "temp_consolidate"
    output_dir = "output_buffer"
    output_path = os.path.join(output_dir, BUFFER_FILE)

    # Clean up any leftover temp dirs
    for d in [temp_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. List only staging. Never call list_repo_files here: the repository
    # has a large legacy root archive which is unrelated to nightly training.
    print("1. Listing staging files in the dataset repo...")
    try:
        all_staging_files = list_staging_files(api)
        staging_files = select_snapshot_files(all_staging_files, cutoff_epoch)
    except Exception as e:
        raise RuntimeError(f"Error listing staging files: {e}") from e

    deferred_count = len(all_staging_files) - len(staging_files)
    print(f"   Snapshot cutoff epoch: {cutoff_epoch}")
    print(f"   Snapshot staging files: {len(staging_files)}")
    print(f"   Files newer than cutoff left for next training: {deferred_count}")

    # 2. Download and read existing replay buffer (if any)
    all_games = []
    existing_buffer_loaded = False
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
        existing_buffer_loaded = True
        print(f"   Read {len(all_games)} games from existing buffer.")
    except Exception as e:
        # The first successful consolidation legitimately has no buffer yet.
        print(f"   No readable existing buffer; continuing with staging: {e}")

    # 3. Download staging files. On the first bootstrap, walk newest-first and
    # stop once the 500k-game replay window is full. Older files are outside
    # the declared sliding window and do not need to be downloaded merely to
    # discard their contents. Incremental runs download every file in the
    # cutoff snapshot, which is expected to be a small set.
    if staging_files:
        print(f"3. Downloading {len(staging_files)} staging files...")
        def download_staging(sf):
            last_error = None
            for attempt in range(1, 6):
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
                except Exception as error:
                    last_error = error
                    if attempt < 5:
                        time.sleep(min(2 ** attempt, 20))
            return sf, [], last_error

        bootstrap = not existing_buffer_loaded
        pending_files = sorted(
            staging_files,
            key=staging_timestamp,
            reverse=bootstrap,
        )
        seen_ids = set()
        for game_str in all_games:
            try:
                seen_ids.add(json.loads(game_str)["id"])
            except Exception:
                pass

        downloaded = 0
        failed = 0
        download_chunk_size = 1000
        for chunk_start in range(0, len(pending_files), download_chunk_size):
            chunk = pending_files[chunk_start:chunk_start + download_chunk_size]
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [executor.submit(download_staging, sf) for sf in chunk]
                for future in as_completed(futures):
                    sf, records, error = future.result()
                    if error is None:
                        all_games.extend(records)
                        downloaded += 1
                        successful_staging_files.append(sf)
                        for record in records:
                            try:
                                seen_ids.add(json.loads(record)["id"])
                            except Exception:
                                pass
                    else:
                        failed += 1
                        if failed <= 5:
                            print(f"   Warning: skipping {sf}: {error}")

            print(
                f"   Progress: {downloaded + failed}/{len(pending_files)} files, "
                f"{len(seen_ids)} unique games collected..."
            )
            if bootstrap and len(seen_ids) >= MAX_REPLAY_GAMES:
                print(
                    "   Bootstrap replay window is full; older staging files "
                    "are outside the 500k-game retention window."
                )
                break

        print(f"   Downloaded {downloaded} files, skipped {failed} files.")
        if failed:
            raise RuntimeError(
                f"Failed to download {failed} of {len(staging_files)} staging files; "
                "refusing to update the replay buffer or delete staging data."
            )
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

    # 5. Enforce both a game-count window and a compressed-byte ceiling. This
    # prevents unusually long records from growing storage back into tens of GB.
    print(f"5. Writing size-bounded buffer ({len(all_games)} candidate games)...")
    all_games, file_size = write_size_bounded_buffer(output_path, all_games)
    file_size_mb = file_size / (1024 * 1024)
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

    # 7. Re-download the committed object before deleting any source data.
    # Self-play may add newer files concurrently; cleanup is restricted to the
    # immutable paths that this run successfully downloaded and consolidated.
    print("7. Verifying committed replay buffer before clearing snapshot files...")
    verified_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=BUFFER_FILE,
        repo_type="dataset",
        token=hf_token,
        local_dir=os.path.join(temp_dir, "verify"),
        force_download=True,
    )
    verified_count = validate_buffer(verified_path, len(all_games))
    print(f"   Verified {verified_count} unique games in the committed buffer.")

    delete_snapshot_files_and_verify(api, successful_staging_files)

    # 8. Cleanup local temp
    for d in [temp_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    print("Done! Replay buffer consolidated successfully.")


if __name__ == '__main__':
    consolidate()
