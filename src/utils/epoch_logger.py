"""Append per-epoch training and validation metrics to CSV files."""

from __future__ import annotations

import csv
import os
from pathlib import Path

_LOG_DIR = Path(os.environ.get("RUN_OUTPUT_ROOT", "/workspace")) / "logs"
_FIELDS = ["fold", "epoch", "train_loss", "val_auroc", "best_val_auroc"]


def log_epoch(
    experiment: str,
    fold: int,
    epoch: int,
    train_loss: float,
    val_auroc: float,
    best_val_auroc: float,
) -> None:
    """Append one fold/epoch record under /workspace/logs by default."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOG_DIR / f"{experiment}_fold{fold}.csv"
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_auroc": f"{val_auroc:.8f}",
                "best_val_auroc": f"{best_val_auroc:.8f}",
            }
        )
