from huggingface_hub import HfApi
import os
try:
    token = os.getenv('HF_TOKEN')
    api = HfApi(token=token)
    repo_id = 'hub-google/DarkChess-NNUE-Data'
    info = api.dataset_info(repo_id, files_metadata=True)
    total_size = sum(f.size for f in info.siblings if f.size)
    print(f'Total size: {total_size / 1024 / 1024:.2f} MB')
    files = [f for f in info.siblings]
    files.sort(key=lambda x: x.size if x.size else 0, reverse=True)
    print("Files:")
    for f in files[:10]:
        size_mb = f.size / 1024 / 1024 if f.size else 0
        print(f"- {f.rfilename}: {size_mb:.2f} MB")
except Exception as e:
    print(f"Error: {e}")
