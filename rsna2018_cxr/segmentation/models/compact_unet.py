"""Compact U-Net used as the trainable in-repo baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CompactUNet(nn.Module):
    """Logits for one foreground class. Smaller than the VGG11 teacher."""

    def __init__(self, in_channels: int = 1, features: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.enc4 = ConvBlock(features * 4, features * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(features * 8, features * 16)
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = ConvBlock(features * 16, features * 8)
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = ConvBlock(features * 8, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        self.outc = nn.Conv2d(features, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.outc(d1)
