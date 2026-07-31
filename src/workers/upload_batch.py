import os
import glob
import gzip
import time
from huggingface_hub import HfApi

REPO_ID = "hub-google/DarkChess-NNUE-Data"


def merge_and_upload():
    output_dir = 'output_data'
    files = sorted(glob.glob(os.path.join(output_dir, '*.jsonl.gz')))
    if not files:
        print("No files to upload.")
        return

    # 1. Merge all local .jsonl.gz files into a single consolidated file
    merged_lines = []
    valid_source_files = []
    for f in files:
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as gz:
                merged_lines.extend(gz.readlines())
            valid_source_files.append(f)
        except Exception as e:
            print(f"Warning: skipping corrupted file {f}: {e}")

    if not merged_lines:
        print("No valid game records found after merging.")
        return

    print(f"Merged {len(merged_lines)} game records from {len(files)} files.")

    # 2. Determine remote path
    path_in_repo = os.environ.get('PATH_IN_REPO')
    if not path_in_repo:
        worker_id = os.environ.get('WORKER_ID', '1')
        date_str = os.environ.get('DATE_STR', time.strftime('%Y%m%d'))
        path_in_repo = f"staging/worker_{worker_id}/{date_str}"

    # 3. Write consolidated file locally
    timestamp = int(time.time())
    consolidated_filename = f'batch_{timestamp}.jsonl.gz'
    consolidated_file = os.path.join(output_dir, consolidated_filename)
    with gzip.open(consolidated_file, 'wt', encoding='utf-8') as gz:
        gz.writelines(merged_lines)

    file_size_kb = os.path.getsize(consolidated_file) / 1024
    print(f"Consolidated file: {consolidated_file} ({file_size_kb:.1f} KB)")

    # 4. Upload single file (NOT upload_folder) to avoid 10k file limit issue
    #    upload_file only touches the one file being uploaded, so it doesn't
    #    trigger HF's per-directory file count check on existing files.
    remote_path = f"{path_in_repo}/{consolidated_filename}"
    print(f"Uploading {consolidated_file} -> {remote_path} ...")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=hf_token)
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=consolidated_file,
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type='dataset',
                commit_message=f"Add batch {consolidated_filename}"
            )
            print(f"Upload complete: {remote_path}")
            break
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'Rate' in error_msg.lower():
                wait = min(60 * attempt, 300)
                print(f"Rate limited (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries:
                print(f"Upload error (attempt {attempt}/{max_retries}): {e}")
                time.sleep(10 * attempt)
            else:
                print(f"Upload failed after {max_retries} attempts: {e}")
                if os.path.exists(consolidated_file):
                    os.remove(consolidated_file)
                raise

    # Delete local source data only after the remote upload succeeds.
    for source_file in valid_source_files:
        if os.path.abspath(source_file) != os.path.abspath(consolidated_file):
            os.remove(source_file)
    os.remove(consolidated_file)


if __name__ == '__main__':
    merge_and_upload()
