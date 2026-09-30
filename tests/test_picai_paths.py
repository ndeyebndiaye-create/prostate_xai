"""
Unit tests for src/data/picai_paths.py — synthetic data only, no dependency
on DATA_ROOT or real patient data, so this runs anywhere including CI.
"""

import pandas as pd
import pytest

from src.data.picai_paths import (
    CSPCA_ISUP_THRESHOLD,
    find_complete_patients,
    load_marksheet,
    picai_image_path,
)


@pytest.fixture
def fake_data_root(tmp_path):
    """
    Build a fake tree matching the confirmed real layout:
    picai/images/<id>/<id>_<study>_000X.mha and
    picai/picai_labels/clinical_information/marksheet.csv
    """
    images_dir = tmp_path / "picai" / "images"
    images_dir.mkdir(parents=True)

    labels_dir = tmp_path / "picai" / "picai_labels" / "clinical_information"
    labels_dir.mkdir(parents=True)

    marksheet = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"],
            "case_ISUP": [2, 4, None, 3],
        }
    )
    marksheet.to_csv(labels_dir / "marksheet.csv", index=False)

    # A: complete (all 3 channels), B: complete, C: missing HBV, D: no folder at all
    for pid, channels in [("A", ["0000", "0001", "0002"]), ("B", ["0000", "0001", "0002"]),
                           ("C", ["0000", "0001"])]:
        pdir = images_dir / pid
        pdir.mkdir()
        for ch in channels:
            (pdir / f"{pid}_study1_{ch}.mha").touch()

    return tmp_path


def test_threshold_is_isup_2():
    assert CSPCA_ISUP_THRESHOLD == 2



def test_find_complete_patients(fake_data_root):
    complete, incomplete = find_complete_patients(fake_data_root)
    assert set(complete) == {"A", "B"}
    assert len(incomplete) == 1
    assert incomplete[0][0] == "C"
    assert incomplete[0][1] == {"hbv"}



def test_load_marksheet_full(fake_data_root):
    df = load_marksheet(fake_data_root)
    # D has no case_ISUP row at all missing? Actually D not in marksheet -> fine
    # C has case_ISUP missing (None) -> dropped
    assert set(df["patient_id"]) == {"A", "B", "D"}
    assert df[df["patient_id"] == "A"]["label"].iloc[0] == 1  # ISUP 2 -> csPCa (threshold >= 2)
    assert df[df["patient_id"] == "B"]["label"].iloc[0] == 1  # ISUP 4 -> csPCa
    assert df[df["patient_id"] == "D"]["label"].iloc[0] == 1  # ISUP 3 -> csPCa



def test_load_marksheet_filtered_to_complete(fake_data_root):
    complete, _ = find_complete_patients(fake_data_root)
    df = load_marksheet(fake_data_root, complete_ids=complete)
    # D is complete=False (no image folder at all) so must be excluded here
    assert set(df["patient_id"]) == {"A", "B"}


def test_picai_image_path_found(fake_data_root):
    path = picai_image_path(fake_data_root, "A", "t2w")
    assert path.name == "A_study1_0000.mha"


def test_picai_image_path_missing_modality_raises(fake_data_root):
    with pytest.raises(FileNotFoundError):
        picai_image_path(fake_data_root, "C", "hbv")  # C is missing HBV channel
