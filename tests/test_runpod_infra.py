from __future__ import annotations

import csv

import torch
from torch import nn

import src.utils.checkpointing as checkpointing
import src.utils.epoch_logger as epoch_logger


def test_training_state_saves_last_best_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpointing, "_CKPT_DIR", tmp_path / "checkpoints")
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    checkpointing.save_training_state(
        model,
        optimizer,
        scaler,
        "experiment",
        1,
        2,
        0.75,
        [0.8],
        [1],
        is_best=True,
    )
    last_path, best_path = checkpointing.fold_checkpoint_paths("experiment", 1)
    assert last_path.is_file()
    assert best_path.is_file()

    new_model = nn.Linear(2, 2)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    new_scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = checkpointing.load_training_state(
        new_model, new_optimizer, new_scaler, "experiment", 1, torch.device("cpu")
    )

    assert state is not None
    assert state["epoch"] == 2
    assert state["best_metric"] == 0.75
    for original, restored in zip(model.parameters(), new_model.parameters()):
        torch.testing.assert_close(original, restored)


def test_epoch_logger_appends_metrics_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(epoch_logger, "_LOG_DIR", tmp_path / "logs")
    epoch_logger.log_epoch("experiment", 1, 1, 0.5, 0.7, 0.7)
    epoch_logger.log_epoch("experiment", 1, 2, 0.4, 0.8, 0.8)

    with (tmp_path / "logs" / "experiment_fold1.csv").open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[-1]["val_auroc"] == "0.80000000"
