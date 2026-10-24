"""Verify that a backed-up best checkpoint can be loaded on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="runpod_backup/checkpoints/fusion_simple/fold_1/best.pt",
        help="Path to best.pt (defaults to the first fusion fold backup).",
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_state = state.get("model_state_dict", state) if isinstance(state, dict) else None
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"No model state dictionary found in {checkpoint}")
    print(f"OK: loaded {checkpoint} on CPU ({len(model_state)} tensors)")


if __name__ == "__main__":
    main()
