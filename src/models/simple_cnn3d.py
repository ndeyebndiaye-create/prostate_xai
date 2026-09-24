"""
Deliberately simple 3D CNN baseline — no skip connections, no attention.

This is the floor of comparison for Step 4 (proposed architecture). If
SegResNet + cross-attention doesn't beat this, the problem is somewhere
other than model complexity (data, labels, preprocessing) — worth knowing
before adding architectural sophistication.
"""

import torch
import torch.nn as nn


class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
