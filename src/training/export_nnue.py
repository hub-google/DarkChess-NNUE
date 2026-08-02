"""Export a trained PyTorch NNUE checkpoint for browser inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train import CURRENT_INPUT_SIZE, load_model_file


TENSORS = (
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
    "fc3.weight", "fc3.bias",
)


def export_model(input_path: Path, output_dir: Path) -> None:
    model = load_model_file(input_path)
    if model.input_size != CURRENT_INPUT_SIZE:
        raise ValueError("only the current 498-input model can be published")
    state = model.state_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "champion.bin"
    metadata = {"format": 1, "inputSize": CURRENT_INPUT_SIZE, "tensors": {}}
    offset = 0
    with binary_path.open("wb") as output:
        for name in TENSORS:
            values = state[name].detach().cpu().numpy().astype("<f4", copy=False)
            raw = values.tobytes(order="C")
            output.write(raw)
            metadata["tensors"][name] = {
                "shape": list(values.shape),
                "offset": offset,
                "length": int(values.size),
            }
            offset += len(raw)
    metadata["bytes"] = offset
    metadata["sha256"] = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    (output_dir / "champion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Exported browser NNUE: {binary_path} ({offset} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/champion.nnue", type=Path)
    parser.add_argument("--output-dir", default="frontend/public/models", type=Path)
    args = parser.parse_args()
    export_model(args.input, args.output_dir)
