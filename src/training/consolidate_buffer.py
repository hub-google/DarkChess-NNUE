import os
import glob
import gzip
import json
import shutil
from huggingface_hub import HfApi, snapshot_download, CommitOperationDelete, CommitOperationAdd

MAX_REPLAY_GAMES = 500_000  # Sliding window capacity (latest 500k games)
REPO_ID = "hub-google/DarkChess-NNUE-Data"
BUFFER_FILE = "replay_buffer.jsonl.gz"

def consolidate():
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    temp_dir = "temp_consolidate"
    output_dir = "output_buffer"
    output_path = os.path.join(output_dir, BUFFER_FILE)

    print("1. Downloading existing dataset from Hugging Face...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=temp_dir, token=hf_token)
    except Exception as e:
        print(f"Warning during snapshot download: {e}")

    # 2. Collect all game records from downloaded files
    all_games = []
    files = glob.glob(os.path.join(temp_dir, "**/*.jsonl.gz"), recursive=True)
    print(f"Discovered {len(files)} dataset files locally. Reading game records...")

    for f in files:
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as gz:
                for line in gz:
                    line_str = line.strip()
                    if line_str:
                        all_games.append(line_str)
        except Exception as e:
            print(f"Skipping corrupted file {f}: {e}")

    print(f"Total game records read: {len(all_games)}")

    # 3. Maintain sliding window up to MAX_REPLAY_GAMES
    if len(all_games) > MAX_REPLAY_GAMES:
        print(f"Game count ({len(all_games)}) exceeds cap. Retaining latest {MAX_REPLAY_GAMES} games.")
        all_games = all_games[-MAX_REPLAY_GAMES:]

    # 4. Pack into a single compressed file
    os.makedirs(output_dir, exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as gz:
        for game_str in all_games:
            gz.write(game_str + '\n')

    print(f"Successfully consolidated into {output_path} with {len(all_games)} games.")

    # 5. Clean HF repo root and upload single replay buffer
    print("5. Updating Hugging Face dataset repo...")
    try:
        online_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
        operations = []
        
        # Delete all existing data files except the new buffer and hidden files
        for of in online_files:
            if not of.startswith(".") and of != BUFFER_FILE:
                operations.append(CommitOperationDelete(path_in_repo=of))

        # Add the consolidated buffer file
        operations.append(CommitOperationAdd(
            path_in_repo=BUFFER_FILE,
            path_or_fileobj=output_path
        ))
        
        # Hugging Face limits ops per commit. Batching by 500 avoids limit.
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
            
        chunks = list(chunker(operations, 500))
        for i, chunk in enumerate(chunks):
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=chunk,
                commit_message=f"Consolidate buffer and cleanup (part {i+1}/{len(chunks)})"
            )
            print(f"Committed chunk {i+1}/{len(chunks)} ({len(chunk)} operations).")

        print("Dataset consolidated successfully on Hugging Face!")
    except Exception as e:
        print(f"Error updating Hugging Face dataset: {e}")

    # Cleanup local temp
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == '__main__':
    consolidate()
