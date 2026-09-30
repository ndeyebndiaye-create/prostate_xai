"""
Multimodal PI-CAI Dataset — loads T2W, ADC, and HBV for a single patient
and stacks them as separate channels: shape (C, D, H, W) where C = len(modalities).

Supports both local filesystem and direct RAM streaming via rclone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.picai_paths import Modality, picai_image_path
from src.data.preprocessing import load_and_normalize_volume

_ALL_MODALITIES: list[Modality] = ["t2w", "adc", "hbv"]


class PicaiMultimodalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        patient_ids: list[str],
        marksheet: pd.DataFrame,
        modalities: list[Modality] | None = None,
        target_shape: tuple[int, int, int] = (96, 96, 16),
        use_rclone_stream: bool = False,
    ):
        if modalities is None:
            modalities = list(_ALL_MODALITIES)
        if not modalities:
            raise ValueError("modalities must be a non-empty list.")
        invalid = set(modalities) - set(_ALL_MODALITIES)
        if invalid:
            raise ValueError(f"Unknown modalities: {invalid}. Valid: {_ALL_MODALITIES}")

        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.modalities = modalities
        self.target_shape = target_shape
        self.use_rclone_stream = use_rclone_stream
        self.labels = marksheet.set_index("patient_id")["label"].to_dict()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patient_ids[idx]

        channels = []
        for modality in self.modalities:
            if self.use_rclone_stream:
                arr = load_and_normalize_volume(
                    patient_id,
                    self.target_shape,
                    modality=modality,
                    use_rclone_stream=True,
                )
            else:
                try:
                    path = picai_image_path(self.data_root, patient_id, modality)
                    arr = load_and_normalize_volume(path, self.target_shape)
                except (FileNotFoundError, Exception):
                    arr = load_and_normalize_volume(
                        patient_id,
                        self.target_shape,
                        modality=modality,
                        use_rclone_stream=True,
                    )
            channels.append(arr)

        volume = np.stack(channels, axis=0)
        tensor = torch.from_numpy(volume)  # (C, D, H, W)
        label = torch.tensor(self.labels[patient_id], dtype=torch.long)
        return tensor, label
