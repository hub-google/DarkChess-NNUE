"""Validate a research NNUE checkpoint before advancing the training lineage."""
import argparse
from pathlib import Path

import torch

from train import CURRENT_INPUT_SIZE, load_model_file


def validate_research_checkpoint(path):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Research checkpoint missing: {checkpoint}")

    model = load_model_file(str(checkpoint))
    if model.input_size != CURRENT_INPUT_SIZE:
        raise ValueError(f"Research checkpoint must have {CURRENT_INPUT_SIZE} inputs")

    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite research model tensor: {name}")

    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(1, CURRENT_INPUT_SIZE))
    if prediction.shape != (1, 1) or not torch.isfinite(prediction).all().item():
        raise ValueError("Non-finite or malformed research model inference")
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    path = validate_research_checkpoint(args.checkpoint)
    print(f"[Research] Valid research NNUE checkpoint: {path}")


if __name__ == "__main__":
    main()
