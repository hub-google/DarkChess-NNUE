"""Non-destructive recovery of a missing DarkChess training dataset.

The original repository returned 404 under its owner's authenticated HF token.
This operation only creates the SAME ID when it is missing. It does not overwrite
or delete any existing dataset. Old removed data are NOT restored by this script.
"""
import os
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

REPO_ID = "hub-google/DarkChess-NNUE-Data"


def ensure_dataset(api, repo_id=REPO_ID, token=None):
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        print(f"[HF Bootstrap] Existing dataset preserved: {repo_id} revision={info.sha}")
        return False
    except RepositoryNotFoundError:
        print(f"[HF Bootstrap] Missing dataset: {repo_id}, creating a fresh private one.")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    info = api.repo_info(repo_id, repo_type="dataset")
    print(f"[HF Bootstrap] CREATED dataset={repo_id}, revision={info.sha}")
    return True


if __name__ == "__main__":
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to create the training dataset")
    api = HfApi(token=token)
    identity = api.whoami(token=token)
    if identity.get("name") != "hub-google":
        raise RuntimeError("HF_TOKEN owner is not hub-google; refusing to create under someone else")
    created = ensure_dataset(api, REPO_ID, token)
    print(f"[HF Bootstrap] new_repository={created}")
