"""A 1D U-Net that segments the borehole into lithologies.

The U-Net comes from image segmentation; here we transplant it to the depth
axis of a well. Encoder/decoder skip connections let the model combine fine
(thin-bed) and coarse (formation-scale) context -- the kind of multi-scale
reasoning a petrophysicist does by eye. Input channels are the standardised
logs *concatenated with their presence mask*, so the network can learn from
which logs exist, not just their values.

Input : (B, 2C, W)   ->   Output: (B, n_classes, W)
W must be divisible by 8 (three 2x downsamples).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .constants import N_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p), nn.BatchNorm1d(c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, k, padding=p), nn.BatchNorm1d(c_out), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = N_CLASSES, base: int = 48):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose1d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv1d(base, n_classes, 1)

    def forward(self, x):
        s1 = self.e1(x)                      # W
        s2 = self.e2(self.pool(s1))          # W/2
        s3 = self.e3(self.pool(s2))          # W/4
        b = self.bottleneck(self.pool(s3))   # W/8
        x = self.d3(torch.cat([self.up3(b), s3], 1))
        x = self.d2(torch.cat([self.up2(x), s2], 1))
        x = self.d1(torch.cat([self.up1(x), s1], 1))
        return self.head(x)                  # (B, n_classes, W)
