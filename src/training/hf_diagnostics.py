"""Read-only dataset / HF token diagnostics. Never print the credential itself."""
import os
from huggingface_hub import HfApi

REPO_ID = os.getenv("HF_DATASET_REPO", "hub-google/DarkChess-NNUE-Data")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN GitHub Secret is absent")
    api = HfApi(token=token)
    try:
        user = api.whoami(token=token)
    except Exception as exc:
        raise RuntimeError("HF_TOKEN does not authenticate; renew secret") from exc
    name = user.get("name", "")
    print("[HF Diagnostic] Authenticated token recognized (value never printed).")
    print(f"[HF Diagnostic] Authenticated owner: {name}")
    candidates = [REPO_ID]
    if name and f"{name}/DarkChess-NNUE-Data" not in candidates:
        candidates.append(f"{name}/DarkChess-NNUE-Data")
    try:
        for d in api.list_datasets(author=name, limit=100):
            identifier = str(getattr(d, "id", ""))
            if "DarkChess" in identifier and identifier not in candidates:
                candidates.append(identifier)
    except Exception as exc:
        print(f"[HF Diagnostic] Cannot list own datasets: {type(exc).__name__}")

    reachable = []
    for candidate in candidates:
        try:
            info = api.repo_info(candidate, repo_type="dataset")
            print(f"[HF Diagnostic] REACHABLE dataset={candidate}, revision={info.sha}")
            reachable.append(candidate)
        except Exception as exc:
            print(f"[HF Diagnostic] NOT FOUND/NOT AUTHORIZED dataset={candidate} ({type(exc).__name__})")

    if not reachable:
        raise RuntimeError("No accessible DarkChess training dataset found. Do not auto-create a fresh empty dataset without deciding how to recover the old replays.")
    print(f"[HF Diagnostic] Accessible DarkChess datasets: {reachable}")


if __name__ == "__main__":
    main()
