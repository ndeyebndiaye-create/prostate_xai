"""
PI-CAI path resolution and label loading — local filesystem version.

Same logic as the earlier Colab version, adapted for two confirmed facts
about the actual downloaded data:
  1. Files are named by nnU-Net channel convention, not by modality name:
     _0000 = T2W, _0001 = ADC, _0002 = HBV.
  2. DATA_ROOT is a normal local path (synced by Google Drive Desktop),
     not a Colab-mounted Drive — no drive.mount() anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

Modality = Literal["t2w", "adc", "hbv"]

CHANNEL_TO_MODALITY = {"0000": "t2w", "0001": "adc", "0002": "hbv"}
MODALITY_TO_CHANNEL = {v: k for k, v in CHANNEL_TO_MODALITY.items()}

# ISUP >= 3 is the EAU-defined threshold for clinically significant PCa (csPCa).
# Fixed here once, project-wide. Do not redefine this elsewhere.
CSPCA_ISUP_THRESHOLD = 3


def picai_image_path(data_root: Path, patient_id: str, modality: Modality) -> Path:
    """
    Resolve the path to a single patient's .mha file for one modality, using
    the confirmed nnU-Net channel naming (_0000/_0001/_0002).

    Raises FileNotFoundError early with a clear message rather than letting
    SimpleITK fail on a bad path later.
    """
    channel = MODALITY_TO_CHANNEL[modality]
    patient_dir = Path(data_root) / "picai" / "images" / str(patient_id)
    matches = list(patient_dir.glob(f"*_{channel}.mha"))
    if not matches:
        raise FileNotFoundError(
            f"No file found for patient={patient_id}, modality={modality} "
            f"(channel {channel}) under {patient_dir}"
        )
    return matches[0]


def find_complete_patients(data_root: Path) -> tuple[list[str], list[tuple[str, set]]]:
    """
    Scan every patient folder under picai/images/ and check that all three
    channels (_0000, _0001, _0002) are present.

    Returns (complete_ids, incomplete) where incomplete is a list of
    (patient_id, missing_channels) — this is the check that produces the
    "how many complete cases do we actually have" number, and it must be
    re-run after every fresh download rather than trusting a remembered count.
    """
    images_root = Path(data_root) / "picai" / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found at {images_root}")

    complete_ids = []
    incomplete = []
    for patient_dir in images_root.iterdir():
        if not patient_dir.is_dir():
            continue
        files = list(patient_dir.glob("*.mha"))
        channels_present = {f.stem.split("_")[-1] for f in files}
        missing = set(CHANNEL_TO_MODALITY.keys()) - channels_present
        if missing:
            incomplete.append((patient_dir.name, missing))
        else:
            complete_ids.append(patient_dir.name)

    return complete_ids, incomplete


def load_marksheet(data_root: Path, complete_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Load picai_marksheet_available.csv, attach the binary csPCa label, and
    optionally filter to only patients confirmed to have complete images
    (via `complete_ids` from find_complete_patients()).

    Rows with a missing case_ISUP are always dropped, with a printed count —
    never silently.
    """
    csv_path = Path(data_root) / "picai" / "picai_labels" / "clinical_information" / "marksheet.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Marksheet not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "case_ISUP" not in df.columns:
        raise ValueError(f"Expected column 'case_ISUP', found: {list(df.columns)}")

    n_before = len(df)
    df = df.dropna(subset=["case_ISUP"]).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[load_marksheet] Dropped {n_dropped} rows with missing case_ISUP.")

    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = (df["case_ISUP"].astype(int) >= CSPCA_ISUP_THRESHOLD).astype(int)

    if complete_ids is not None:
        n_before_filter = len(df)
        df = df[df["patient_id"].isin(set(complete_ids))].copy()
        print(
            f"[load_marksheet] Filtered to complete-image patients: "
            f"{n_before_filter} -> {len(df)}"
        )

    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(
        f"[load_marksheet] Final: {len(df)} patients "
        f"(label=0: {n_neg}, label=1: {n_pos}, ratio {n_neg / max(n_pos, 1):.2f}:1)"
    )
    return df
