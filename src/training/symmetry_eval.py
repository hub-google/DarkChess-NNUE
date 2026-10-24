"""Measure value-model errors on all eight exact Banqi state symmetries.

Diagnostic only: the paired-match SPRT remains the sole promotion gate.
No extra self-play data or HF writes are required. Run after cache creation:
python src/training/symmetry_eval.py --cache-dir training_cache \
  --model models/champion.nnue --model models/challenger.nnue \
  --samples 256 --output symmetry_report.json
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from train import CURRENT_INPUT_SIZE, SYMMETRY_COUNT, feature_permutation, load_model_file


def load_audit_positions(cache_dir, samples=256, seed=20261024):
    """Draw a bounded, reproducible sample across at most 16 cache shards."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    paths = sorted(glob.glob(str(Path(cache_dir) / "cache_*.npz")))
    if not paths:
        raise RuntimeError(f"no training cache shards in {cache_dir}")
    rng = np.random.default_rng(seed)
    selected = [paths[int(i)] for i in np.linspace(0, len(paths) - 1, min(16, len(paths)), dtype=int)]
    per_file = math.ceil(samples / len(selected))
    rows = []
    for path in selected:
        with np.load(path) as shard:
            features = shard["features"]
            if not len(features):
                continue
            indices = rng.choice(len(features), size=min(len(features), per_file), replace=False)
            rows.append(features[indices].astype(np.float32))
    if not rows:
        raise RuntimeError("training cache contains no audit positions")
    output = np.concatenate(rows, axis=0)[:samples]
    if output.shape[1] != CURRENT_INPUT_SIZE:
        raise ValueError("audit expects 498-channel public-state features")
    return output


def symmetry_metrics(model, features):
    """Return 7 transform error pairs (mean/max abs) on Red-perspective outputs."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != model.input_size:
        raise ValueError("audit sample width and model input_size must match")
    model.eval()
    torch.set_num_threads(1)
    x = torch.from_numpy(features)
    details = []
    with torch.no_grad():
        baseline = model(x).reshape(-1)
        for swap in (False, True):
            for transform in range(SYMMETRY_COUNT):
                if transform == 0 and not swap:
                    continue
                indices = torch.as_tensor(
                    feature_permutation(model.input_size, transform, swap),
                    dtype=torch.long
                )
                predictions = model(x.index_select(1, indices)).reshape(-1)
                differences = (baseline - (-predictions if swap else predictions)).abs()
                details.append({
                    "transform": transform,
                    "color_swap": swap,
                    "mean_abs_error": float(differences.mean().item()),
                    "max_abs_error": float(differences.max().item()),
                })
    return {
        "positions": len(features),
        "overall_mean_abs_error": sum(row["mean_abs_error"] for row in details) / len(details),
        "geometry_mean_abs_error": sum(row["mean_abs_error"] for row in details if not row["color_swap"]) / 3,
        "color_mean_abs_error": sum(row["mean_abs_error"] for row in details if row["color_swap"]) / 4,
        "transforms": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit exact 4x8 board/color symmetry")
    parser.add_argument("--cache-dir", default="training_cache")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", default="symmetry_report.json")
    args = parser.parse_args()
    features = load_audit_positions(args.cache_dir, args.samples, args.seed)
    report = {"samples": len(features), "seed": args.seed, "models": {}}
    for path in args.model:
        name = Path(path).name
        if not Path(path).is_file():
            report["models"][name] = {"status": "missing"}
            continue
        model = load_model_file(path)
        if model.input_size != CURRENT_INPUT_SIZE:
            report["models"][name] = {"status": "legacy_input_size", "input_size": model.input_size}
            continue
        report["models"][name] = symmetry_metrics(model, features)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
