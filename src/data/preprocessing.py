"""
Shared preprocessing utilities for PI-CAI volume loading and normalization.

These functions are the single source of truth for all preprocessing logic
used across unimodal, multimodal, and proposed-architecture datasets.

Includes support for zero-disk-footprint streaming directly from Google Drive
via rclone in RAM without saving full datasets to local storage.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODALITY_TO_TOKENS = {
    "t2w": ["t2w", "0000"],
    "adc": ["adc", "0001"],
    "hbv": ["hbv", "0002"],
}


def stream_volume_from_drive(
    patient_id: str,
    modality: str,
    rclone_remote: str = "gdrive_thesis:thesis_data",
) -> np.ndarray:
    """
    Stream a single patient volume directly from Google Drive into RAM via rclone cat.
    Leaves ZERO files on the local disk.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            res = subprocess.run(
                ["rclone", "lsf", f"{rclone_remote}/picai/images/{patient_id}/"],
                capture_output=True,
                text=True,
                check=True,
            )
            break
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0 * (2 ** attempt))

    files = [f.strip() for f in res.stdout.strip().split("\n") if f.strip()]

    tokens = MODALITY_TO_TOKENS.get(modality, [modality])
    matched_file = None
    for f in files:
        for t in tokens:
            if f.endswith(f"_{t}.mha"):
                matched_file = f
                break
        if matched_file:
            break

    if not matched_file:
        raise FileNotFoundError(
            f"No file found for patient={patient_id}, modality={modality} on remote {rclone_remote}"
        )

    remote_file_path = f"{rclone_remote}/picai/images/{patient_id}/{matched_file}"

    for attempt in range(max_retries):
        try:
            cat_res = subprocess.run(
                ["rclone", "cat", remote_file_path],
                capture_output=True,
                check=True,
            )
            break
        except subprocess.CalledProcessError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0 * (2 ** attempt))

    with tempfile.NamedTemporaryFile(suffix=".mha") as tmp:
        tmp.write(cat_res.stdout)
        tmp.flush()
        img = sitk.ReadImage(tmp.name)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)

    return arr


def load_and_normalize_volume(
    path_or_patient: Path | str,
    target_shape: tuple[int, int, int],
    modality: str | None = None,
    use_rclone_stream: bool = False,
    rclone_remote: str = "gdrive_thesis:thesis_data",
) -> np.ndarray:
    """
    Load a .mha volume (from local path or streamed directly from Google Drive in RAM),
    apply masked z-score normalization, clip to ±3σ, and center-crop/pad to `target_shape`.
    """
    if use_rclone_stream:
        patient_id = str(path_or_patient)
        if modality is None:
            raise ValueError("modality is required when use_rclone_stream=True")
        arr = stream_volume_from_drive(patient_id, modality, rclone_remote=rclone_remote)
    else:
        path = Path(path_or_patient)
        if not path.exists():
            # Fallback to streaming if local path is not present
            if modality is not None:
                patient_id = path.parent.name
                arr = stream_volume_from_drive(patient_id, modality, rclone_remote=rclone_remote)
            else:
                raise FileNotFoundError(f"File not found locally at {path}")
        else:
            img = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # Masked z-score: exclude background (zero-intensity) pixels
    mask = arr != 0
    if mask.sum() > 0:
        mean, std = arr[mask].mean(), arr[mask].std()
        arr = (arr - mean) / (std + 1e-8)
    arr = np.clip(arr, -3.0, 3.0)

    arr = center_crop_or_pad(arr, target_shape)
    return arr


def center_crop_or_pad(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Deterministically center-crop (if array exceeds target) or zero-pad
    (if array is smaller) a 3-D volume to exactly `target_shape`.
    """
    target_h, target_w, target_d = target_shape
    d, h, w = arr.shape

    def _bounds(size: int, target: int) -> tuple[slice, int, int]:
        if size >= target:
            start = (size - target) // 2
            return slice(start, start + target), 0, 0
        pad_before = (target - size) // 2
        pad_after = target - size - pad_before
        return slice(0, size), pad_before, pad_after

    d_slice, d_pb, d_pa = _bounds(d, target_d)
    h_slice, h_pb, h_pa = _bounds(h, target_h)
    w_slice, w_pb, w_pa = _bounds(w, target_w)

    cropped = arr[d_slice, h_slice, w_slice]
    return np.pad(cropped, ((d_pb, d_pa), (h_pb, h_pa), (w_pb, w_pa)))
