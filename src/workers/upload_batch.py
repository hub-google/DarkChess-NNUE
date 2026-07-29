import os
import glob
import gzip
import time
import sys
from huggingface_hub import HfApi

def merge_and_upload():
    output_dir = 'output_data'
    files = glob.glob(os.path.join(output_dir, '*.jsonl.gz'))
    if not files:
        print("No files to upload.")
        return

    merged_lines = []
    for f in files:
        with gzip.open(f, 'rt', encoding='utf-8') as gz:
            merged_lines.extend(gz.readlines())
        os.remove(f)

    timestamp = int(time.time())
    
    # Path inside repo: staging/worker_{WORKER_ID}/{DATE_STR} for self_play, staging/fresh for train
    path_in_repo = os.environ.get('PATH_IN_REPO')
    if not path_in_repo:
        worker_id = os.environ.get('WORKER_ID', '1')
        date_str = os.environ.get('DATE_STR', time.strftime('%Y%m%d'))
        path_in_repo = f"staging/worker_{worker_id}/{date_str}"

    consolidated_file = os.path.join(output_dir, f'batch_{timestamp}.jsonl.gz')
    with gzip.open(consolidated_file, 'wt', encoding='utf-8') as gz:
        gz.writelines(merged_lines)
    
    print(f"Uploading consolidated batch {consolidated_file} to {path_in_repo} ...")
    HfApi().upload_folder(
        folder_path=output_dir,
        repo_id='hub-google/DarkChess-NNUE-Data',
        repo_type='dataset',
        path_in_repo=path_in_repo
    )
    os.remove(consolidated_file)
    print("Upload complete.")

if __name__ == '__main__':
    merge_and_upload()
