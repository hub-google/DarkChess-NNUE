"""
One-time cleanup script: delete ALL files under staging/ on Hugging Face
to fix the 10,000 file per directory limit error.

After this runs, the staging directories will be empty and workers can
resume uploading normally.
"""
import os
import time
from huggingface_hub import HfApi, CommitOperationDelete

REPO_ID = "hub-google/DarkChess-NNUE-Data"
CHUNK_SIZE = 500  # HF recommends max ~500 operations per commit


def cleanup_staging():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set.")
        return

    api = HfApi(token=hf_token)

    print(f"Listing all files in {REPO_ID}...")
    try:
        all_files = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return

    # Find all staging files (these are the scattered data_*.jsonl.gz files)
    staging_files = [f for f in all_files if f.startswith("staging/")]
    print(f"Total files in repo: {len(all_files)}")
    print(f"Staging files to delete: {len(staging_files)}")

    if not staging_files:
        print("No staging files to clean up. Done!")
        return

    # Delete in chunks to avoid API limits
    chunks = [staging_files[i:i + CHUNK_SIZE] for i in range(0, len(staging_files), CHUNK_SIZE)]
    print(f"Will delete in {len(chunks)} commits of up to {CHUNK_SIZE} files each.")

    for i, chunk in enumerate(chunks):
        delete_ops = [CommitOperationDelete(path_in_repo=f) for f in chunk]
        for attempt in range(1, 6):
            try:
                api.create_commit(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    operations=delete_ops,
                    commit_message=f"Cleanup staging files (batch {i+1}/{len(chunks)}, {len(chunk)} files)"
                )
                print(f"  Deleted batch {i+1}/{len(chunks)} ({len(chunk)} files)")
                break
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'rate' in error_msg.lower():
                    wait = 60 * attempt
                    print(f"  Rate limited on batch {i+1}. Waiting {wait}s (attempt {attempt}/5)...")
                    time.sleep(wait)
                elif attempt < 5:
                    print(f"  Error on batch {i+1} (attempt {attempt}/5): {e}")
                    time.sleep(15 * attempt)
                else:
                    print(f"  FAILED batch {i+1} after 5 attempts: {e}")

        # Small delay between commits to avoid rate limits
        if i < len(chunks) - 1:
            time.sleep(5)

    # Verify
    remaining = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    remaining_staging = [f for f in remaining if f.startswith("staging/")]
    print(f"\nCleanup complete. Remaining staging files: {len(remaining_staging)}")
    print(f"Total files in repo now: {len(remaining)}")


if __name__ == '__main__':
    cleanup_staging()
