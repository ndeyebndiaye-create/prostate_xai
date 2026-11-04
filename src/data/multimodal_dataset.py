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
CLINICAL_FEATURE_NAMES = ["patient_age", "log_psa", "psad", "prostate_volume"]


def fit_transform_clinical_features(
    marksheet: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    eps: float = 1e-3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract and standardize the 4 clinical features defined in Chapter 3, Section 3.2.2:
        [patient_age, log(PSA + eps), PSAD, prostate_volume]
    Imputation (median/mean) and z-score parameters (mean, std) are computed STRICTLY
    on `train_ids` and reused without modification on `val_ids` to prevent data leakage
    (Section 3.2.2 & Section 3.5).

    Returns
    -------
    (train_clin_dict, val_clin_dict, stats_dict)
    where each clin_dict maps patient_id -> np.ndarray of shape (4,) (float32).
    """
    df = marksheet.drop_duplicates(subset=["patient_id"], keep="first").set_index("patient_id")

    def _raw_table(pids: list[str]) -> pd.DataFrame:
        sub = df.reindex(pids)
        age = pd.to_numeric(sub.get("patient_age"), errors="coerce")
        psa = pd.to_numeric(sub.get("psa"), errors="coerce")
        vol = pd.to_numeric(sub.get("prostate_volume"), errors="coerce")
        psad = pd.to_numeric(sub.get("psad"), errors="coerce")
        # Derive PSAD = PSA / volume when missing and both PSA and volume are valid
        derived_psad = psa / vol.replace(0, np.nan)
        psad = psad.fillna(derived_psad)
        log_psa = np.log(np.clip(psa, a_min=0.0, a_max=None) + eps)
        return pd.DataFrame(
            {
                "patient_age": age,
                "log_psa": log_psa,
                "psad": psad,
                "prostate_volume": vol,
            },
            index=pids,
        )

    train_df = _raw_table(train_ids)
    val_df = _raw_table(val_ids)

    # Fit imputation (train means) and standardization (train mean & std) on train ONLY
    train_means = train_df.mean(axis=0).fillna(0.0)
    train_filled = train_df.fillna(train_means)
    val_filled = val_df.fillna(train_means)

    mean_vec = train_filled.mean(axis=0).to_numpy(dtype=np.float32)
    std_vec = train_filled.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std_vec = np.where(std_vec < 1e-8, 1.0, std_vec)

    train_norm = (train_filled.to_numpy(dtype=np.float32) - mean_vec) / std_vec
    val_norm = (val_filled.to_numpy(dtype=np.float32) - mean_vec) / std_vec

    train_dict = {pid: train_norm[i] for i, pid in enumerate(train_ids)}
    val_dict = {pid: val_norm[i] for i, pid in enumerate(val_ids)}
    stats = {"mean": mean_vec, "std": std_vec, "feature_names": CLINICAL_FEATURE_NAMES}
    return train_dict, val_dict, stats


class PicaiMultimodalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        patient_ids: list[str],
        marksheet: pd.DataFrame,
        modalities: list[Modality] | None = None,
        target_shape: tuple[int, int, int] = (96, 96, 16),
        use_rclone_stream: bool = False,
        align_to_t2w: bool = False,
        clinical_features: dict[str, np.ndarray] | None = None,
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
        self.align_to_t2w = align_to_t2w
        self.clinical_features = clinical_features
        self.labels = marksheet.set_index("patient_id")["label"].to_dict()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        patient_id = self.patient_ids[idx]

        ref_t2w_path: Path | None = None
        if self.align_to_t2w and not self.use_rclone_stream:
            try:
                ref_t2w_path = picai_image_path(self.data_root, patient_id, "t2w")
            except FileNotFoundError:
                ref_t2w_path = None

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
                    ref_path = ref_t2w_path if (self.align_to_t2w and modality != "t2w") else None
                    arr = load_and_normalize_volume(
                        path,
                        self.target_shape,
                        reference_path=ref_path,
                    )
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

        if self.clinical_features is not None:
            clin_arr = self.clinical_features[patient_id]
            clin_tensor = torch.from_numpy(np.asarray(clin_arr, dtype=np.float32))
            return tensor, clin_tensor, label

        return tensor, label
