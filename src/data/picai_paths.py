"""
PI-CAI path resolution and label loading — robust dual naming & RAM streaming support.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import pandas as pd

Modality = Literal["t2w", "adc", "hbv"]

CHANNEL_TO_MODALITY = {
    "0000": "t2w",
    "0001": "adc",
    "0002": "hbv",
    "t2w": "t2w",
    "adc": "adc",
    "hbv": "hbv",
}

MODALITY_TO_TOKENS = {
    "t2w": ["t2w", "0000"],
    "adc": ["adc", "0001"],
    "hbv": ["hbv", "0002"],
}

# ISUP >= 2 (Gleason >= 3+4) is the clinically significant PCa (csPCa) threshold.
CSPCA_ISUP_THRESHOLD = 2


def picai_image_path(data_root: Path, patient_id: str, modality: Modality) -> Path:
    """
    Resolve the path to a single patient's .mha file for one modality, supporting
    both literal modality suffixes (_t2w, _adc, _hbv) and nnU-Net channel numbers (_0000, _0001, _0002).
    """
    patient_dir = Path(data_root) / "picai" / "images" / str(patient_id)
    tokens = MODALITY_TO_TOKENS[modality]
    matches = []
    for token in tokens:
        matches.extend(list(patient_dir.glob(f"*_{token}.mha")))
    if not matches:
        raise FileNotFoundError(
            f"No file found for patient={patient_id}, modality={modality} under {patient_dir}"
        )
    return matches[0]


def find_complete_patients(
    data_root: Path,
    rclone_remote: str = "gdrive_thesis:thesis_data",
) -> tuple[list[str], list[tuple[str, set]]]:
    """
    Scan patient folders under picai/images/ to return all complete patient IDs.
    If local directory is absent, queries Google Drive via rclone lsf.
    """
    images_root = Path(data_root) / "picai" / "images"
    complete_ids = []
    incomplete = []
    required_modalities = {"t2w", "adc", "hbv"}

    if images_root.exists():
        for patient_dir in images_root.iterdir():
            if not patient_dir.is_dir():
                continue
            files = list(patient_dir.glob("*.mha"))
            found = set()
            for f in files:
                stem_suffix = f.stem.split("_")[-1]
                if stem_suffix in CHANNEL_TO_MODALITY:
                    found.add(CHANNEL_TO_MODALITY[stem_suffix])
            missing = required_modalities - found
            if missing:
                incomplete.append((patient_dir.name, missing))
            else:
                complete_ids.append(patient_dir.name)
    else:
        try:
            res = subprocess.run(
                ["rclone", "lsf", f"{rclone_remote}/picai/images/", "--dirs-only"],
                capture_output=True,
                text=True,
                check=True,
            )
            dirs = [d.strip("/") for d in res.stdout.strip().split("\n") if d.strip()]
            complete_ids = sorted(dirs)
        except Exception:
            pass

    return sorted(complete_ids), incomplete


def load_marksheet(data_root: Path, complete_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Load picai_marksheet_available.csv, attach the binary csPCa label, and
    optionally filter to only patients confirmed to have complete images.
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

    # Déduplication au niveau patient (un patient peut avoir plusieurs examens dans le CSV).
    # On trie par 'label' et 'case_ISUP' décroissants pour garder l'examen le plus sévère en cas de doublon.
    n_before_dedup = len(df)
    df = df.sort_values(by=["label", "case_ISUP"], ascending=False)
    df = df.drop_duplicates(subset=["patient_id"], keep="first").copy()
    n_dedup = n_before_dedup - len(df)
    if n_dedup > 0:
        print(f"[load_marksheet] Deduplicated {n_dedup} secondary study rows for unique patient-level splitting.")

    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(
        f"[load_marksheet] Final: {len(df)} patients "
        f"(label=0: {n_neg}, label=1: {n_pos}, ratio {n_neg / max(n_pos, 1):.2f}:1)"
    )
    return df