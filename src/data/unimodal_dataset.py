"""
Unimodal PI-CAI Dataset (T2W, ADC, or HBV) — supports local files and RAM streaming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.picai_paths import Modality, picai_image_path
from src.data.preprocessing import load_and_normalize_volume


class PicaiUnimodalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        patient_ids: list[str],
        marksheet: pd.DataFrame,
        modality: Modality = "t2w",
        target_shape: tuple[int, int, int] = (96, 96, 16),
        use_rclone_stream: bool = False,
    ):
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.modality = modality
        self.target_shape = target_shape
        self.use_rclone_stream = use_rclone_stream
        self.labels = marksheet.set_index("patient_id")["label"].to_dict()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patient_ids[idx]
        if self.use_rclone_stream:
            arr = load_and_normalize_volume(
                patient_id,
                self.target_shape,
                modality=self.modality,
                use_rclone_stream=True,
            )
        else:
            try:
                path = picai_image_path(self.data_root, patient_id, self.modality)
                arr = load_and_normalize_volume(path, self.target_shape)
            except (FileNotFoundError, Exception):
                arr = load_and_normalize_volume(
                    patient_id,
                    self.target_shape,
                    modality=self.modality,
                    use_rclone_stream=True,
                )

        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)
        label = torch.tensor(self.labels[patient_id], dtype=torch.long)
        return tensor, label
