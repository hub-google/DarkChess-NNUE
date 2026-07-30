"""
Consolidate Replay Buffer

Downloads all staging .jsonl.gz files from the HF dataset repo,
merges them into a single replay_buffer.jsonl.gz (capped at 500k games),
uploads the buffer, and deletes the old staging files.

Uses list_repo_files + hf_hub_download (file-by-file) instead of
snapshot_download to avoid choking on directories with 10000+ files.
"""
import os
import gzip
import json
import shutil
import time
from huggingface_hub import HfApi, hf_hub_download, CommitOperationDelete

MAX_REPLAY_GAMES = 500_000  # Sliding window capacity
REPO_ID = "hub-google/DarkChess-NNUE-Data"
BUFFER_FILE = "replay_buffer.jsonl.gz"
CHUNK_SIZE = 500  # Max delete operations per commit


def consolidate():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        return

    api = HfApi(token=hf_token)
    temp_dir = "temp_consolidate"
    output_dir = "output_buffer"
    output_path = os.path.join(output_dir, BUFFER_FILE)

    # Clean up any leftover temp dirs
    for d in [temp_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. List all files in the repo
    print("1. Listing all files in the dataset repo...")
    try:
        all_files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return

    # Separate into: existing replay buffer + staging files
    jsonl_files = [f for f in all_files if f.endswith('.jsonl.gz')]
    staging_files = [f for f in jsonl_files if f.startswith("staging/")]
    has_existing_buffer = BUFFER_FILE in all_files

    print(f"   Total files: {len(all_files)}")
    print(f"   Existing replay buffer: {'Yes' if has_existing_buffer else 'No'}")
    print(f"   Staging .jsonl.gz files: {len(staging_files)}")

    # 2. Download and read existing replay buffer (if any)
    all_games = []
    if has_existing_buffer:
        print("2. Downloading existing replay buffer...")
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
            print(f"   Warning: could not read existing buffer: {e}")

    # 3. Download staging files one by one
    if staging_files:
        print(f"3. Downloading {len(staging_files)} staging files...")
        downloaded = 0
        failed = 0
        for i, sf in enumerate(staging_files):
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID, filename=sf,
                    repo_type="dataset", token=hf_token, local_dir=temp_dir
                )
                with gzip.open(local_path, 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        line_str = line.strip()
                        if line_str:
                            all_games.append(line_str)
                downloaded += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"   Warning: skipping {sf}: {e}")

            # Progress every 50 files
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(staging_files)} files processed, accumulated {len(all_games)} games so far...")

        print(f"   Downloaded {downloaded} files, skipped {failed} files.")
    else:
        print("3. No staging files to process.")

    print(f"   Total game records collected: {len(all_games)}")

    if not all_games:
        print("No games found. Nothing to consolidate.")
        return

    # 4. Apply sliding window cap
    if len(all_games) > MAX_REPLAY_GAMES:
        print(f"4. Trimming from {len(all_games)} to latest {MAX_REPLAY_GAMES} games.")
        all_games = all_games[-MAX_REPLAY_GAMES:]
    else:
        print(f"4. Game count ({len(all_games)}) within cap, keeping all.")

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
                print(f"   FATAL: Upload failed after 5 attempts: {e}")
                return

    # 7. Delete staging files from HF repo
    if staging_files:
        print(f"7. Cleaning up {len(staging_files)} staging files from HF...")
        chunks = [staging_files[i:i + CHUNK_SIZE] for i in range(0, len(staging_files), CHUNK_SIZE)]

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
