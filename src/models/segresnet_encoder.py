"""
Modality-specific 3D SegResNet Encoder and Clinical MLP Encoder
(Chapter 4, Sections 4.2 and 4.3 of the thesis manuscript).

- `SegResNetEncoder3D` wraps MONAI's 3D `SegResNet` backbone (`spatial_dims=3`,
  `init_filters=16` or `32`, `blocks_down=(1, 2, 2, 4)`), strips the unused
  segmentation decoder head, stores the deepest 3D convolutional feature map
  (`self.last_feature_map`) for independent per-modality 3D Grad-CAM++
  attribution (Section 4.6.1), and pools/projects to a dense representation.
- `ClinicalEncoderMLP` encodes the 4 structured clinical features
  `[patient_age, log(PSA), PSAD, prostate_volume]` using `LayerNorm + GELU + Dropout`
  for stability with small 3D batches (Section 4.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet


class SegResNetEncoder3D(nn.Module):
    """
    3D SegResNet encoder for a single MRI sequence (T2W, ADC, or HBV).

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for unimodal encoder branch).
    init_filters : int
        Initial number of convolutional filters (16 or 32, Section 3.10 & 4.2).
    blocks_down : tuple[int, ...]
        Number of residual blocks per encoder stage, default `(1, 2, 2, 4)`.
    out_dim : int
        Output embedding dimension after global average pooling and projection
        (default 256 as in Section 4.2, or 128 for direct token projection).
    dropout : float | None
        Optional dropout probability inside SegResNet blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        init_filters: int = 16,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        out_dim: int = 256,
        dropout: float | None = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.out_dim = out_dim

        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=2,
            dropout_prob=dropout,
            blocks_down=blocks_down,
            blocks_up=(1,) * (len(blocks_down) - 1),
        )
        # Remove unused segmentation decoder layers (Section 4.2:
        # "La tête de segmentation native de SegResNet est supprimée")
        self.backbone.up_layers = nn.ModuleList()
        self.backbone.up_samples = nn.ModuleList()
        self.backbone.conv_final = nn.Identity()

        deep_channels = init_filters * (2 ** (len(blocks_down) - 1))
        self.deep_channels = deep_channels
        self.pool = nn.AdaptiveAvgPool3d(1)

        if deep_channels == out_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(deep_channels, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        # Populated on every forward pass for Grad-CAM++ (Section 4.6.1)
        self.last_feature_map: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deepest 3D convolutional feature map `[B, C_deep, d, h, w]`."""
        latent, _ = self.backbone.encode(x)
        if latent.requires_grad:
            latent.retain_grad()
        self.last_feature_map = latent
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `[B, 1, D, H, W]` -> `[B, out_dim]`."""
        fmap = self.forward_features(x)
        pooled = self.pool(fmap).flatten(1)
        return self.proj(pooled)


class ClinicalEncoderMLP(nn.Module):
    """
    2-layer MLP with LayerNorm and GELU for the 4 clinical variables
    `[patient_age, log(PSA), PSAD, prostate_volume]` (Chapter 4, Section 4.3):

        [B, 4] -> Linear(4, 64) -> LayerNorm + GELU + Dropout
               -> Linear(64, 64) -> LayerNorm + GELU
    """

    def __init__(
        self,
        in_features: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, clin: torch.Tensor) -> torch.Tensor:
        """Encode `[B, 4]` -> `[B, hidden_dim]` (default 64)."""
        return self.net(clin)
