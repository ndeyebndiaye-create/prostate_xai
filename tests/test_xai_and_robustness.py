"""
Unit tests for Steps 6, 7, and 8 (Calibration, XAI, Uncertainty, and Low-Resource Degradations):
- `src/utils/calibration_metrics.py` (sens90 threshold, Youden threshold, ECE, Brier)
- `src/xai/gradcam3d.py` (per-modality 3D Grad-CAM++, Pointing Accuracy, Occlusion Fidelity)
- `src/xai/shap_clinical.py` (exact Shapley values on the clinical branch logit + additivity check)
- `src/xai/mc_dropout.py` (Monte Carlo Dropout + selective abstention curve)
- `src/data/degradation.py` (Rician noise, 3D Gaussian blur, 1.5T axial downsampling)
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.degradation import (
    apply_axial_downsampling_3d,
    apply_gaussian_blur_3d,
    apply_rician_noise_3d,
    degrade_multimodal_volume,
)
from src.models.cross_attention_fusion import MultimodalCrossAttentionNet
from src.utils.calibration_metrics import (
    compute_clinical_and_calibration_metrics,
    compute_expected_calibration_error,
)
from src.xai.gradcam3d import (
    compute_occlusion_fidelity_drop,
    compute_per_modality_gradcam_pp_3d,
    compute_pointing_accuracy,
)
from src.xai.mc_dropout import evaluate_selective_abstention, mc_dropout_predict
from src.xai.shap_clinical import compute_exact_clinical_shap_on_logit


def test_calibration_and_clinical_thresholds():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95])
    m_sens90 = compute_clinical_and_calibration_metrics(
        labels, probs, threshold_strategy="sens90"
    )
    assert m_sens90["auroc"] == 1.0
    assert m_sens90["sensitivity"] >= 0.90
    assert 0.0 <= m_sens90["ece"] <= 1.0
    assert 0.0 <= m_sens90["brier"] <= 1.0

    ece = compute_expected_calibration_error(labels, probs, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_per_modality_gradcam3d_and_occlusion_and_pointing():
    model = MultimodalCrossAttentionNet(
        modalities=["t2w", "adc", "hbv"],
        init_filters=8,
        encoder_dim=64,
        fusion_dim=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        use_clinical=True,
        fusion_mode="cross_attention",
    )
    x = torch.randn(1, 3, 16, 32, 32)
    c = torch.randn(1, 4)

    heatmaps = compute_per_modality_gradcam_pp_3d(model, x, clin=c)
    assert set(heatmaps.keys()) == {"t2w", "adc", "hbv"}
    for mod, hmap in heatmaps.items():
        assert hmap.shape == (16, 32, 32), f"Unexpected shape for {mod}: {hmap.shape}"
        assert hmap.min() >= 0.0 and hmap.max() <= 1.0 + 1e-5

    # Pointing accuracy test
    fake_mask = np.zeros((16, 32, 32), dtype=np.uint8)
    z_max, y_max, x_max = np.unravel_index(
        int(np.argmax(heatmaps["t2w"])), heatmaps["t2w"].shape
    )
    fake_mask[z_max, y_max, x_max] = 1
    assert compute_pointing_accuracy(heatmaps["t2w"], fake_mask) is True

    drops = compute_occlusion_fidelity_drop(model, x, heatmaps, clin=c)
    assert "drop_t2w" in drops and "drop_joint" in drops


def test_exact_clinical_shap_additivity_on_logit():
    model = MultimodalCrossAttentionNet(
        modalities=["t2w", "adc", "hbv"],
        init_filters=8,
        encoder_dim=64,
        fusion_dim=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        use_clinical=True,
        fusion_mode="cross_attention",
    )
    x = torch.randn(1, 3, 16, 32, 32)
    c = torch.tensor([[1.2, -0.5, 2.1, -0.8]], dtype=torch.float32)

    shap_vals = compute_exact_clinical_shap_on_logit(model, x, c)
    phi_sum = (
        shap_vals["patient_age"]
        + shap_vals["log_psa"]
        + shap_vals["psad"]
        + shap_vals["prostate_volume"]
    )
    logit_diff = shap_vals["full_logit"] - shap_vals["base_logit"]
    assert np.isclose(phi_sum, logit_diff, atol=1e-4)


def test_mc_dropout_and_selective_abstention():
    model = MultimodalCrossAttentionNet(
        modalities=["t2w", "adc", "hbv"],
        init_filters=8,
        encoder_dim=64,
        fusion_dim=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        dropout=0.25,
        use_clinical=False,
    )
    x = torch.randn(4, 3, 16, 32, 32)
    out = mc_dropout_predict(model, x, n_passes=10)
    assert out["mean_prob"].shape == (4,)
    assert out["var_prob"].shape == (4,)
    assert out["all_probs"].shape == (10, 4)

    curve = evaluate_selective_abstention(
        labels=[0, 1, 0, 1, 0, 1],
        mean_probs=[0.1, 0.9, 0.2, 0.8, 0.49, 0.51],
        uncertainties=[0.01, 0.01, 0.02, 0.02, 0.25, 0.25],
    )
    assert len(curve) == 6
    assert curve[0]["n_retained"] == 6


def test_degradation_operators_preserve_shape():
    vol = np.random.randn(16, 32, 32).astype(np.float32)
    r = apply_rician_noise_3d(vol, alpha=0.06)
    b = apply_gaussian_blur_3d(vol, sigma=1.0)
    d = apply_axial_downsampling_3d(vol, inplane_factor=2, slice_factor=2)
    assert r.shape == vol.shape
    assert b.shape == vol.shape
    assert d.shape == vol.shape

    m_vol = np.random.randn(3, 16, 32, 32).astype(np.float32)
    for sev in ("clean", "low", "medium", "high"):
        deg = degrade_multimodal_volume(m_vol, severity=sev)  # type: ignore[arg-type]
        assert deg.shape == m_vol.shape
