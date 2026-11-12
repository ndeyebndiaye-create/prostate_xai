"""
Per-Modality 3D Grad-CAM++, Pointing Accuracy, and Occlusion Fidelity
(Chapter 4, Section 4.6.1 of the thesis manuscript).

Key design rule (Section 4.6.1):
- Each MRI modality (`T2W`, `ADC`, `HBV`) produces its OWN independent 3D
  saliency volume (`Heatmap_T2W`, `Heatmap_ADC`, `Heatmap_HBV`) computed from
  `model.last_feature_maps[modality]`. The three maps are NEVER collapsed into
  a single undifferentiated heatmap so radiologists can inspect anatomical (T2W)
  vs diffusion (ADC/HBV) activations separately.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_per_modality_gradcam_pp_3d(
    model: nn.Module,
    images: torch.Tensor,
    clin: torch.Tensor | None = None,
    target_class: int = 1,
    upsample_to_input: bool = True,
    percentile_threshold: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Compute independent 3D Grad-CAM++ heatmaps for every modality in `model.modalities`.

    Formulation (Chattopadhay et al., 2018 / Section 4.6.1):
        Let Y^c be the logit for class `target_class` (default 1 = csPCa) and
        A^k_{d,h,w} be the k-th channel of the deepest 3D feature map of modality m.
        With g = dY^c / dA^k:
            alpha^k_{d,h,w} = g^2 / (2 * g^2 + sum_{a,b,c} A^k_{a,b,c} * g^3 + eps)
            w_k = sum_{d,h,w} alpha^k_{d,h,w} * ReLU(g_{d,h,w})
            L_m = ReLU( sum_k w_k * A^k )

    Parameters
    ----------
    model : MultimodalCrossAttentionNet
        Trained model with `model.modalities` and `model.last_feature_maps`.
    images : torch.Tensor of shape `[1, C, D, H, W]`
        Single patient 3D multimodal input tensor.
    clin : optional torch.Tensor of shape `[1, 4]`
        Clinical feature vector if `model.use_clinical=True`.
    target_class : int
        Target logit index (1 for csPCa positive class).
    upsample_to_input : bool
        If True, trilinearly upsamples each 3D heatmap to `(D, H, W)` and min-max
        normalizes to `[0, 1]`.
    percentile_threshold : float
        Optional percentile threshold (e.g. 70.0) to filter out background diffuse
        noise and sharpen focal tumor localization. Default is 0.0 (no thresholding).
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping `modality -> 3D numpy array of shape (D, H, W)` in `[0, 1]`.
    """
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    x = images.detach().clone().to(device).requires_grad_(True)
    c = clin.detach().clone().to(device) if clin is not None else None

    model.zero_grad(set_to_none=True)
    logits = model(x, clin=c)
    score = logits[:, target_class].sum()
    score.backward(retain_graph=False)

    _, _, in_d, in_h, in_w = x.shape
    heatmaps: dict[str, np.ndarray] = {}

    for modality in getattr(model, "modalities", ["t2w", "adc", "hbv"]):
        fmap = model.last_feature_maps.get(modality)
        if fmap is None or fmap.grad is None:
            heatmaps[modality] = np.zeros((in_d, in_h, in_w), dtype=np.float32)
            continue

        A = fmap.detach()  # [1, K, d, h, w]
        g = fmap.grad.detach()  # [1, K, d, h, w]

        g2 = g ** 2
        g3 = g ** 3
        spatial_sum_Ag3 = (A * g3).sum(dim=(2, 3, 4), keepdim=True)
        denom = 2.0 * g2 + spatial_sum_Ag3 + eps
        alpha = g2 / denom
        weights = (alpha * F.relu(g)).sum(dim=(2, 3, 4), keepdim=True)  # [1, K, 1, 1, 1]

        cam = F.relu((weights * A).sum(dim=1, keepdim=True))  # [1, 1, d, h, w]
        if upsample_to_input:
            cam = F.interpolate(
                cam,
                size=(in_d, in_h, in_w),
                mode="trilinear",
                align_corners=False,
            )
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max - c_min > eps:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        if percentile_threshold > 0:
            p_val = float(np.percentile(cam_np, percentile_threshold))
            cam_np = np.where(cam_np >= p_val, cam_np, 0.0)
            c_min2, c_max2 = float(cam_np.min()), float(cam_np.max())
            if c_max2 - c_min2 > eps:
                cam_np = (cam_np - c_min2) / (c_max2 - c_min2)

        heatmaps[modality] = cam_np

    if was_training:
        model.train()
    return heatmaps


def compute_pointing_accuracy(
    heatmap_3d: np.ndarray,
    lesion_mask_3d: np.ndarray,
    tolerance_voxels: int = 1,
) -> bool:
    """
    Pointing Game Accuracy (Section 4.6.1):
    Returns True (hit = 1) if the 3D coordinate `(z*, y*, x*)` of maximum
    activation in `heatmap_3d` falls inside the binary ground-truth `lesion_mask_3d`
    (with an optional 3D neighborhood tolerance of `tolerance_voxels` to account
    for coarse feature map resolution before trilinear upsampling).
    """
    if lesion_mask_3d.sum() <= 0:
        return False
    max_idx = np.unravel_index(int(np.argmax(heatmap_3d)), heatmap_3d.shape)
    z, y, x = max_idx
    d, h, w = lesion_mask_3d.shape
    z0, z1 = max(0, z - tolerance_voxels), min(d, z + tolerance_voxels + 1)
    y0, y1 = max(0, y - tolerance_voxels), min(h, y + tolerance_voxels + 1)
    x0, x1 = max(0, x - tolerance_voxels), min(w, x + tolerance_voxels + 1)
    return bool(np.any(lesion_mask_3d[z0:z1, y0:y1, x0:x1] > 0))


def compute_occlusion_fidelity_drop(
    model: nn.Module,
    images: torch.Tensor,
    heatmaps: dict[str, np.ndarray],
    clin: torch.Tensor | None = None,
    top_k_fraction: float = 0.05,
) -> dict[str, float]:
    """
    Occlusion Fidelity Test (Section 4.6.1):
    Masks out the top `top_k_fraction` (default 5%) most salient voxels in each
    modality (setting them to background 0.0 in z-score space) and measures the
    drop in predicted csPCa probability:
        Delta_prob = P(csPCa | original) - P(csPCa | occluded)
    """
    model.eval()
    device = next(model.parameters()).device
    x = images.detach().clone().to(device)
    c = clin.detach().clone().to(device) if clin is not None else None

    with torch.no_grad():
        base_logits = model(x, clin=c)
        base_prob = float(torch.softmax(base_logits, dim=1)[0, 1].item())

    drops: dict[str, float] = {}
    modalities = list(getattr(model, "modalities", ["t2w", "adc", "hbv"]))

    # Per-modality occlusion drop
    for idx, mod in enumerate(modalities):
        hmap = heatmaps.get(mod)
        if hmap is None:
            continue
        thresh = float(np.quantile(hmap, 1.0 - top_k_fraction))
        mask_bool = torch.from_numpy(hmap >= thresh).to(device)
        x_occ = x.clone()
        x_occ[0, idx][mask_bool] = 0.0
        with torch.no_grad():
            occ_prob = float(torch.softmax(model(x_occ, clin=c), dim=1)[0, 1].item())
        drops[f"drop_{mod}"] = base_prob - occ_prob

    # Joint top-salient occlusion across all modalities
    x_joint = x.clone()
    for idx, mod in enumerate(modalities):
        hmap = heatmaps.get(mod)
        if hmap is not None:
            thresh = float(np.quantile(hmap, 1.0 - top_k_fraction))
            mask_bool = torch.from_numpy(hmap >= thresh).to(device)
            x_joint[0, idx][mask_bool] = 0.0
    with torch.no_grad():
        joint_prob = float(torch.softmax(model(x_joint, clin=c), dim=1)[0, 1].item())
    drops["drop_joint"] = base_prob - joint_prob
    drops["base_prob"] = base_prob
    return drops
