"""
Unit tests for:
  - src/data/preprocessing.py  (load_and_normalize_volume, center_crop_or_pad)
  - src/data/unimodal_dataset.py  (refactored — regression check)
  - src/data/multimodal_dataset.py  (PicaiMultimodalDataset)
  - src/utils/results_logger.py  (save_results)

All tests use synthetic data only. No dependency on DATA_ROOT, real PI-CAI
images, or any external filesystem mount. Tests can run in any environment,
including CI, without network access.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch

from src.data.preprocessing import center_crop_or_pad, load_and_normalize_volume
from src.data.multimodal_dataset import PicaiMultimodalDataset
from src.utils.results_logger import save_results


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_fake_mha(path: Path, shape: tuple[int, int, int] = (20, 64, 64)) -> None:
    """Write a synthetic .mha volume at the given path with random voxel values."""
    rng = np.random.default_rng(seed=42)
    arr = rng.uniform(0, 1000, size=shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, str(path))


@pytest.fixture
def fake_mha(tmp_path) -> Path:
    """Single .mha file with a deterministic random volume (20, 64, 64)."""
    p = tmp_path / "test_volume.mha"
    _write_fake_mha(p, shape=(20, 64, 64))
    return p


@pytest.fixture
def fake_data_root_multimodal(tmp_path) -> tuple[Path, pd.DataFrame]:
    """
    Minimal fake PI-CAI tree supporting PicaiMultimodalDataset.

    Structure:
      picai/images/P001/P001_study1_0000.mha  (T2W)
      picai/images/P001/P001_study1_0001.mha  (ADC)
      picai/images/P001/P001_study1_0002.mha  (HBV)
      picai/images/P002/...  (same)
    """
    images_dir = tmp_path / "picai" / "images"

    patient_ids = ["P001", "P002"]
    for pid in patient_ids:
        pdir = images_dir / pid
        pdir.mkdir(parents=True)
        for ch in ["0000", "0001", "0002"]:
            _write_fake_mha(pdir / f"{pid}_study1_{ch}.mha", shape=(10, 32, 32))

    marksheet = pd.DataFrame({
        "patient_id": patient_ids,
        "label": [0, 1],
    })
    return tmp_path, marksheet


# ---------------------------------------------------------------------------
# Tests: preprocessing.py
# ---------------------------------------------------------------------------


class TestCenterCropOrPad:
    def test_no_op_when_already_target_shape(self):
        arr = np.ones((16, 96, 96), dtype=np.float32)
        result = center_crop_or_pad(arr, target_shape=(96, 96, 16))
        assert result.shape == (16, 96, 96)

    def test_crops_large_volume(self):
        arr = np.zeros((32, 128, 128), dtype=np.float32)
        result = center_crop_or_pad(arr, target_shape=(96, 96, 16))
        assert result.shape == (16, 96, 96)

    def test_pads_small_volume(self):
        arr = np.zeros((8, 48, 48), dtype=np.float32)
        result = center_crop_or_pad(arr, target_shape=(96, 96, 16))
        assert result.shape == (16, 96, 96)

    def test_mixed_crop_and_pad(self):
        # D=8 < 16 (pad), H=128 > 96 (crop), W=48 < 96 (pad)
        arr = np.zeros((8, 128, 48), dtype=np.float32)
        result = center_crop_or_pad(arr, target_shape=(96, 96, 16))
        assert result.shape == (16, 96, 96)

    def test_values_preserved_after_center_crop(self):
        # Only the center portion should be retained.
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[3:7, 3:7, 3:7] = 1.0  # center block
        result = center_crop_or_pad(arr, target_shape=(4, 4, 4))
        assert result.shape == (4, 4, 4)
        # The center of arr is all-ones, so output should be all-ones.
        assert np.allclose(result, 1.0)

    def test_output_dtype_preserved(self):
        arr = np.zeros((16, 96, 96), dtype=np.float64)
        result = center_crop_or_pad(arr, target_shape=(96, 96, 16))
        assert result.dtype == np.float64


class TestLoadAndNormalizeVolume:
    def test_output_shape_matches_target(self, fake_mha, tmp_path):
        result = load_and_normalize_volume(fake_mha, target_shape=(48, 48, 12))
        assert result.shape == (12, 48, 48)

    def test_output_dtype_is_float32(self, fake_mha):
        result = load_and_normalize_volume(fake_mha, target_shape=(64, 64, 16))
        assert result.dtype == np.float32

    def test_values_clipped_to_3_sigma(self, fake_mha):
        result = load_and_normalize_volume(fake_mha, target_shape=(64, 64, 16))
        assert result.min() >= -3.0 - 1e-6
        assert result.max() <= 3.0 + 1e-6

    def test_all_zero_volume_does_not_crash(self, tmp_path):
        """A completely black volume (e.g. failed scan) must not raise."""
        p = tmp_path / "zero.mha"
        arr = np.zeros((10, 32, 32), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, str(p))
        result = load_and_normalize_volume(p, target_shape=(32, 32, 10))
        assert result.shape == (10, 32, 32)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# Tests: multimodal_dataset.py
# ---------------------------------------------------------------------------


class TestPicaiMultimodalDataset:
    def test_three_channel_output_shape(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        ds = PicaiMultimodalDataset(
            data_root,
            patient_ids=["P001"],
            marksheet=marksheet,
            modalities=["t2w", "adc", "hbv"],
            target_shape=(32, 32, 10),
        )
        tensor, label = ds[0]
        assert tensor.shape == (3, 10, 32, 32), f"Expected (3,10,32,32), got {tensor.shape}"

    def test_two_channel_output_shape(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        ds = PicaiMultimodalDataset(
            data_root,
            patient_ids=["P001"],
            marksheet=marksheet,
            modalities=["t2w", "adc"],
            target_shape=(32, 32, 10),
        )
        tensor, label = ds[0]
        assert tensor.shape == (2, 10, 32, 32), f"Expected (2,10,32,32), got {tensor.shape}"

    def test_label_is_correct(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        ds = PicaiMultimodalDataset(
            data_root,
            patient_ids=["P001", "P002"],
            marksheet=marksheet,
            modalities=["t2w", "adc", "hbv"],
            target_shape=(32, 32, 10),
        )
        _, label0 = ds[0]
        _, label1 = ds[1]
        assert label0.item() == 0
        assert label1.item() == 1

    def test_tensor_dtype_is_float32(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        ds = PicaiMultimodalDataset(
            data_root,
            patient_ids=["P001"],
            marksheet=marksheet,
            modalities=["t2w"],
            target_shape=(32, 32, 10),
        )
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32

    def test_invalid_modality_raises(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        with pytest.raises(ValueError, match="Unknown modalities"):
            PicaiMultimodalDataset(
                data_root,
                patient_ids=["P001"],
                marksheet=marksheet,
                modalities=["t2w", "flair"],  # "flair" is invalid
            )

    def test_empty_modalities_raises(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        with pytest.raises(ValueError, match="non-empty"):
            PicaiMultimodalDataset(
                data_root,
                patient_ids=["P001"],
                marksheet=marksheet,
                modalities=[],
            )

    def test_len_matches_patient_ids(self, fake_data_root_multimodal):
        data_root, marksheet = fake_data_root_multimodal
        ds = PicaiMultimodalDataset(
            data_root,
            patient_ids=["P001", "P002"],
            marksheet=marksheet,
            modalities=["t2w", "adc", "hbv"],
            target_shape=(32, 32, 10),
        )
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# Tests: results_logger.py
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_creates_csv_with_header_and_rows(self, tmp_path, monkeypatch):
        """save_results must create the CSV and write n_folds + 1 rows."""
        import src.utils.results_logger as rl

        # Patch the module-level path constants to use tmp_path.
        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        fold_aurocs = [0.70, 0.72, 0.68, 0.71, 0.73]
        save_results("test_experiment", "t2w", fold_aurocs, config={})

        csv_path = tmp_path / "results" / "baselines.csv"
        assert csv_path.exists()

        rows = list(csv.DictReader(csv_path.open()))
        # 5 per-fold rows + 1 summary row
        assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"

    def test_summary_row_has_correct_fold_label(self, tmp_path, monkeypatch):
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        save_results("test_exp", "adc", [0.65, 0.67, 0.63], config={})

        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        summary_rows = [r for r in rows if r["fold"] == "summary"]
        assert len(summary_rows) == 1

    def test_mean_auroc_is_correct(self, tmp_path, monkeypatch):
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        fold_aurocs = [0.60, 0.80]
        save_results("test_exp2", "hbv", fold_aurocs, config={})

        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        summary = next(r for r in rows if r["fold"] == "summary")
        assert abs(float(summary["mean_auroc"]) - 0.70) < 1e-4

    def test_appends_across_multiple_calls(self, tmp_path, monkeypatch):
        """Two calls must produce two summary rows (append, not overwrite)."""
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        save_results("exp_a", "t2w", [0.70, 0.71], config={})
        save_results("exp_b", "adc", [0.65, 0.66], config={})

        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        summary_rows = [r for r in rows if r["fold"] == "summary"]
        assert len(summary_rows) == 2


# ---------------------------------------------------------------------------
# Tests: seed.py
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_same_seed_produces_same_tensor(self):
        """Deux appels set_seed(42) doivent produire des tenseurs identiques."""
        from src.utils.seed import set_seed

        set_seed(42)
        t1 = torch.randn(5, 5)
        set_seed(42)
        t2 = torch.randn(5, 5)
        assert torch.allclose(t1, t2), "set_seed ne garantit pas la reproductibilité torch"

    def test_different_seeds_produce_different_tensors(self):
        """Deux seeds différentes doivent donner des résultats différents."""
        from src.utils.seed import set_seed

        set_seed(42)
        t1 = torch.randn(5, 5)
        set_seed(99)
        t2 = torch.randn(5, 5)
        assert not torch.allclose(t1, t2), "Des seeds différentes donnent le même résultat"

    def test_numpy_seed_is_fixed(self):
        """La seed numpy est fixée indépendamment de torch."""
        from src.utils.seed import set_seed

        set_seed(7)
        a1 = np.random.rand(10)
        set_seed(7)
        a2 = np.random.rand(10)
        assert np.allclose(a1, a2), "set_seed ne fixe pas numpy correctement"


# ---------------------------------------------------------------------------
# Tests: fold_guard.py
# ---------------------------------------------------------------------------


class TestFoldGuard:
    def test_no_overlap_passes_silently(self):
        """Pas d'overlap → aucune exception."""
        from src.utils.fold_guard import assert_no_patient_overlap

        assert_no_patient_overlap(["A", "B", "C"], ["D", "E", "F"])  # ne lève rien

    def test_overlap_raises_runtime_error(self):
        """Un patient dans les deux sets → RuntimeError."""
        from src.utils.fold_guard import assert_no_patient_overlap

        with pytest.raises(RuntimeError, match="FUITE"):
            assert_no_patient_overlap(["A", "B", "C"], ["C", "D", "E"])

    def test_empty_train_no_error(self):
        """Train vide → pas d'overlap possible."""
        from src.utils.fold_guard import assert_no_patient_overlap

        assert_no_patient_overlap([], ["A", "B"])

    def test_class_distribution_prints(self, capsys):
        """log_fold_class_distribution doit afficher les counts corrects."""
        from src.utils.fold_guard import log_fold_class_distribution

        log_fold_class_distribution([0, 1, 0, 0, 1], fold=1, split="val")
        captured = capsys.readouterr()
        assert "fold 1" in captured.out
        assert "label=0: 3" in captured.out
        assert "label=1: 2" in captured.out


# ---------------------------------------------------------------------------
# Tests: training_utils.py
# ---------------------------------------------------------------------------


class TestTrainingUtils:
    def test_confusion_matrix_perfect_prediction(self):
        """Prédiction parfaite → TP=2, TN=2, FP=0, FN=0."""
        from src.utils.training_utils import log_fold_confusion_matrix

        labels = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.9, 0.8]
        result = log_fold_confusion_matrix(labels, probs, threshold=0.5)
        assert result["tp"] == 2
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert abs(result["sensitivity"] - 1.0) < 1e-4
        assert abs(result["specificity"] - 1.0) < 1e-4

    def test_confusion_matrix_all_predicted_negative(self):
        """Modèle qui prédit toujours 0 → sensibilité ≈ 0."""
        from src.utils.training_utils import log_fold_confusion_matrix

        labels = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.4]  # tous < 0.5
        result = log_fold_confusion_matrix(labels, probs, threshold=0.5)
        assert result["tp"] == 0
        assert result["fn"] == 2
        assert result["sensitivity"] < 0.01

    def test_confusion_matrix_has_required_keys(self):
        """Le dict de retour doit contenir toutes les clés attendues."""
        from src.utils.training_utils import log_fold_confusion_matrix

        result = log_fold_confusion_matrix([0, 1], [0.3, 0.7])
        for key in ["tn", "fp", "fn", "tp", "sensitivity", "specificity"]:
            assert key in result, f"Clé manquante : {key}"


# ---------------------------------------------------------------------------
# Tests: results_logger.py — colonnes de reproductibilité
# ---------------------------------------------------------------------------


class TestSaveResultsEnriched:
    def test_git_hash_column_written(self, tmp_path, monkeypatch):
        """La colonne git_hash est écrite dans le CSV."""
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        save_results(
            "test_exp", "t2w", [0.70, 0.72],
            config={"git_hash": "abc1234", "seed": 42, "config_file": "configs/phase1_t2w.yaml"},
        )
        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        assert "git_hash" in rows[0]
        assert rows[0]["git_hash"] == "abc1234"

    def test_seed_column_written(self, tmp_path, monkeypatch):
        """La colonne seed est écrite dans le CSV."""
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        save_results("exp_seed", "adc", [0.65], config={"seed": 99})
        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        assert rows[0]["seed"] == "99"

    def test_missing_repro_fields_use_defaults(self, tmp_path, monkeypatch):
        """Si git_hash/seed/config_file absents du config, valeurs par défaut."""
        import src.utils.results_logger as rl

        monkeypatch.setattr(rl, "_RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr(rl, "_CSV_PATH", tmp_path / "results" / "baselines.csv")

        save_results("exp_default", "hbv", [0.60], config={})
        rows = list(csv.DictReader((tmp_path / "results" / "baselines.csv").open()))
        assert rows[0]["git_hash"] == "unknown"
        assert rows[0]["config_file"] == "argparse"
