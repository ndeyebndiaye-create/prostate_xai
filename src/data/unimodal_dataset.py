"""
Unimodal PI-CAI Dataset (T2W, ADC, or HBV) — local filesystem version.

No ROI mask cropping yet (deliberate — see README Step scaffolding): this is
the simple baseline stage, so we use a fixed center-crop/resize. Mask-based
cropping is deferred to the proposed-architecture step, once prostate
segmentation masks are confirmed available for all complete cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from src.data.picai_paths import Modality, picai_image_path


class PicaiUnimodalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        patient_ids: list[str],
        marksheet: pd.DataFrame,
        modality: Modality = "t2w",
        target_shape: tuple[int, int, int] = (96, 96, 16),
    ):
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.modality = modality
        self.target_shape = target_shape
        self.labels = marksheet.set_index("patient_id")["label"].to_dict()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_and_preprocess(self, patient_id: str) -> np.ndarray:
        path = picai_image_path(self.data_root, patient_id, self.modality)
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)

        # Masked z-score: background (air) pixels excluded from mean/std,
        # otherwise the large zero-background region drags the statistics.
        mask = arr != 0
        if mask.sum() > 0:
            mean, std = arr[mask].mean(), arr[mask].std()
            arr = (arr - mean) / (std + 1e-8)
        arr = np.clip(arr, -3, 3)

        arr = self._center_crop_or_pad(arr, self.target_shape)
        return arr

    @staticmethod
    def _center_crop_or_pad(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        target_h, target_w, target_d = target_shape
        d, h, w = arr.shape

        def bounds(size: int, target: int) -> tuple[slice, int, int]:
            if size >= target:
                start = (size - target) // 2
                return slice(start, start + target), 0, 0
            pad_before = (target - size) // 2
            pad_after = target - size - pad_before
            return slice(0, size), pad_before, pad_after

        d_slice, d_pb, d_pa = bounds(d, target_d)
        h_slice, h_pb, h_pa = bounds(h, target_h)
        w_slice, w_pb, w_pa = bounds(w, target_w)

        cropped = arr[d_slice, h_slice, w_slice]
        return np.pad(cropped, ((d_pb, d_pa), (h_pb, h_pa), (w_pb, w_pa)))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id = self.patient_ids[idx]
        arr = self._load_and_preprocess(patient_id)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)
        label = torch.tensor(self.labels[patient_id], dtype=torch.long)
        return tensor, label
