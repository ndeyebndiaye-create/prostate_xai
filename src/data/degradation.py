"""
Controlled Low-Resource Degradation Operators (Chapter 3, Section 3.4 & Chapter 4, Section 4.5).

Simulates African clinical acquisition conditions on 3D multiparametric MRI:
1. **Rician noise 3D** on T2W and HBV sequences:
   I_noisy = sqrt((I + n1)^2 + n2^2), where n1, n2 ~ N(0, (alpha * sigma_Omega)^2)
   with alpha in {0.03 (low), 0.06 (medium), 0.10 (high)}.
2. **3D Gaussian blur** on ADC:
   sigma in {0.5 (low), 1.0 (medium), 1.5 (high)} voxels.
3. **Axial downsampling (1.5T low-field simulation)**:
   Downsamples spatial resolution by factor 2 or 3 followed by trilinear interpolation.
4. **Multimodal degradation coordinator (`degrade_multimodal_volume`)**:
   Applies coordinated perturbations across sequences according to severity levels:
   `clean`, `low`, `medium`, `high`.
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

SeverityLevel = Literal["clean", "low", "medium", "high"]

# Severity mappings defined in Chapter 3, Section 3.4
SEVERITY_RICIAN_ALPHA: dict[SeverityLevel, float] = {
    "clean": 0.0,
    "low": 0.03,
    "medium": 0.06,
    "high": 0.10,
}

SEVERITY_GAUSSIAN_SIGMA: dict[SeverityLevel, float] = {
    "clean": 0.0,
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}


def apply_rician_noise_3d(
    volume: np.ndarray,
    alpha: float = 0.06,
    seed: int | None = None,
) -> np.ndarray:
    """
    Applies 3D Rician noise to an MRI volume:
    I_noisy = sqrt((I + n1)^2 + n2^2), where n1, n2 ~ N(0, (alpha * sigma_Omega)^2).

    Args:
        volume: 3D float numpy array (D, H, W).
        alpha: Noise scale factor relative to volume standard deviation.
        seed: Optional RNG seed.
    """
    if alpha <= 0.0:
        return volume.copy()

    rng = np.random.default_rng(seed)
    sigma_omega = float(np.std(volume))
    if sigma_omega < 1e-8:
        sigma_omega = 1.0

    noise_std = alpha * sigma_omega
    n1 = rng.normal(0.0, noise_std, size=volume.shape).astype(volume.dtype)
    n2 = rng.normal(0.0, noise_std, size=volume.shape).astype(volume.dtype)

    noisy = np.sqrt(np.maximum((volume + n1) ** 2 + n2**2, 0.0)).astype(volume.dtype)
    return noisy


def apply_gaussian_blur_3d(
    volume: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Applies separable 3D Gaussian blur to an MRI volume (e.g., ADC).

    Args:
        volume: 3D float numpy array (D, H, W).
        sigma: Standard deviation in voxels for Gaussian kernel.
    """
    if sigma <= 0.0:
        return volume.copy()
    blurred = gaussian_filter(volume, sigma=sigma, mode="nearest")
    return blurred.astype(volume.dtype)


def apply_axial_downsampling_3d(
    volume: np.ndarray,
    inplane_factor: int = 2,
    slice_factor: int = 2,
) -> np.ndarray:
    """
    Simulates lower-field (1.5T) acquisition with thicker slices and lower in-plane resolution.
    Downsamples the volume then interpolates back to original spatial shape (D, H, W).

    Args:
        volume: 3D float numpy array with shape (D, H, W).
        inplane_factor: Downsampling factor along H and W.
        slice_factor: Downsampling factor along axial depth D.
    """
    if inplane_factor <= 1 and slice_factor <= 1:
        return volume.copy()

    d, h, w = volume.shape
    down_factors = (1.0 / slice_factor, 1.0 / inplane_factor, 1.0 / inplane_factor)
    downsampled = zoom(volume, down_factors, order=1, mode="nearest")

    up_factors = (d / downsampled.shape[0], h / downsampled.shape[1], w / downsampled.shape[2])
    restored = zoom(downsampled, up_factors, order=1, mode="nearest")

    # Match exact shape in case of rounding differences
    if restored.shape != volume.shape:
        out = np.zeros_like(volume)
        min_d = min(d, restored.shape[0])
        min_h = min(h, restored.shape[1])
        min_w = min(w, restored.shape[2])
        out[:min_d, :min_h, :min_w] = restored[:min_d, :min_h, :min_w]
        return out.astype(volume.dtype)

    return restored.astype(volume.dtype)


def degrade_multimodal_volume(
    multimodal_volume: np.ndarray,
    modalities: Sequence[str] = ("t2w", "adc", "hbv"),
    severity: SeverityLevel = "medium",
    downsample_15t: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """
    Coordinates degradation across multimodal MRI sequences:
    - T2W: Rician noise (+ optional 1.5T axial downsampling)
    - ADC: 3D Gaussian blur (+ optional 1.5T axial downsampling)
    - HBV: Rician noise (+ optional 1.5T axial downsampling)

    Args:
        multimodal_volume: 4D float array (C, D, H, W).
        modalities: Names of modalities corresponding to channel dimension C.
        severity: One of `clean`, `low`, `medium`, `high`.
        downsample_15t: If True, also applies axial downsampling to simulate 1.5T.
        seed: Optional RNG seed.
    """
    if severity == "clean":
        return multimodal_volume.copy()

    rician_alpha = SEVERITY_RICIAN_ALPHA[severity]
    blur_sigma = SEVERITY_GAUSSIAN_SIGMA[severity]

    out = multimodal_volume.copy()
    for idx, mod in enumerate(modalities):
        vol_3d = out[idx]
        mod_seed = (seed + idx * 100) if seed is not None else None

        if mod.lower() in ("t2w", "hbv"):
            vol_3d = apply_rician_noise_3d(vol_3d, alpha=rician_alpha, seed=mod_seed)
        elif mod.lower() == "adc":
            vol_3d = apply_gaussian_blur_3d(vol_3d, sigma=blur_sigma)

        if downsample_15t:
            slice_fac = 2 if severity in ("low", "medium") else 3
            inplane_fac = 1 if severity == "low" else 2
            vol_3d = apply_axial_downsampling_3d(vol_3d, inplane_factor=inplane_fac, slice_factor=slice_fac)

        out[idx] = vol_3d

    return out
