"""
Unit tests for Step 4 & Step 5 modules (Chapter 3.3 & Chapter 4 of the thesis):
- Physical-space resampling (`resample_to_reference`)
- Leak-free clinical feature standardization (`fit_transform_clinical_features`)
- `SegResNetEncoder3D` & `ClinicalEncoderMLP`
- `MultimodalCrossAttentionNet` (Model-U vs Model-S, C5/C6/C7, ModalityGating, availability_mask)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from src.data.multimodal_dataset import fit_transform_clinical_features
from src.data.preprocessing import resample_to_reference
from src.models.cross_attention_fusion import MultimodalCrossAttentionNet
from src.models.segresnet_encoder import ClinicalEncoderMLP, SegResNetEncoder3D


def test_resample_to_reference_aligns_geometry():
    ref = sitk.Image(64, 64, 16, sitk.sitkFloat32)
    ref.SetSpacing((0.5, 0.5, 3.0))
    ref.SetOrigin((-16.0, -16.0, 0.0))

    mov = sitk.Image(32, 30, 16, sitk.sitkFloat32)
    mov.SetSpacing((1.5, 1.5, 3.0))
    mov.SetOrigin((-20.0, -20.0, 0.0))

    out = resample_to_reference(mov, ref)
    assert out.GetSize() == ref.GetSize()
    assert np.allclose(out.GetSpacing(), ref.GetSpacing())
    assert np.allclose(out.GetOrigin(), ref.GetOrigin())


def test_fit_transform_clinical_features_no_leakage():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "patient_age": [60.0, 70.0, 65.0, 100.0],
            "psa": [4.0, 10.0, 8.0, 500.0],
            "psad": [0.10, np.nan, 0.20, 5.0],
            "prostate_volume": [40.0, 50.0, 40.0, 100.0],
            "label": [0, 1, 0, 1],
        }
    )
    train_ids = ["p1", "p2", "p3"]
    val_ids = ["p4"]

    train_clin, val_clin, stats = fit_transform_clinical_features(df, train_ids, val_ids)
    assert set(train_clin.keys()) == set(train_ids)
    assert set(val_clin.keys()) == set(val_ids)
    assert train_clin["p1"].shape == (4,)
    # Train features must have zero mean across train_ids (excluding extreme val p4)
    stacked_train = np.stack([train_clin[p] for p in train_ids], axis=0)
    assert np.allclose(stacked_train.mean(axis=0), 0.0, atol=1e-5)
    assert not np.isnan(stacked_train).any()
    assert not np.isnan(val_clin["p4"]).any()


def test_segresnet_encoder_3d_shapes_and_feature_map():
    enc = SegResNetEncoder3D(in_channels=1, init_filters=16, out_dim=256)
    x = torch.randn(2, 1, 16, 48, 48)
    out = enc(x)
    assert out.shape == (2, 256)
    assert enc.last_feature_map is not None
    assert enc.last_feature_map.shape == (2, 128, 2, 6, 6)


def test_clinical_encoder_mlp():
    mlp = ClinicalEncoderMLP(in_features=4, hidden_dim=64)
    c = torch.randn(3, 4)
    out = mlp(c)
    assert out.shape == (3, 64)


def test_multimodal_cross_attention_step4_image_only():
    model = MultimodalCrossAttentionNet(
        modalities=["t2w", "adc", "hbv"],
        init_filters=8,
        encoder_dim=128,
        fusion_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128,
        shared_weights=False,
        use_clinical=False,
        fusion_mode="cross_attention",
        use_gating=True,
    )
    x = torch.randn(2, 3, 16, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 2)
    assert model.last_gate_weights is not None
    assert model.last_gate_weights.shape == (2, 3)
    assert set(model.last_feature_maps.keys()) == {"t2w", "adc", "hbv"}

    loss = logits.sum()
    loss.backward()
    for m in ["t2w", "adc", "hbv"]:
        grads = [
            p.grad for p in model.encoders[m].parameters() if p.grad is not None
        ]
        assert len(grads) > 0


def test_multimodal_cross_attention_step5_c5_c6_c7_and_mask():
    x = torch.randn(2, 3, 16, 32, 32)
    clin = torch.randn(2, 4)

    for mode in ("concat", "attention", "cross_attention"):
        model = MultimodalCrossAttentionNet(
            modalities=["t2w", "adc", "hbv"],
            init_filters=8,
            encoder_dim=128,
            fusion_dim=64,
            num_heads=4,
            num_layers=1,
            ffn_dim=128,
            shared_weights=(mode == "attention"),  # also test Model-S
            use_clinical=True,
            fusion_mode=mode,  # type: ignore[arg-type]
            use_gating=True,
        )
        # Drop HBV (index 2) for patient 0 via availability_mask
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        logits = model(x, clin=clin, availability_mask=mask)
        assert logits.shape == (2, 2)
        assert model.last_gate_weights is not None
        assert model.last_gate_weights.shape == (2, 4)
        # Dropped modality (index 2 of sample 0) must receive ~0 gate weight
        assert float(model.last_gate_weights[0, 2].item()) < 1e-3
