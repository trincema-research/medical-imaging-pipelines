"""U-Net with a VGG11-BN encoder.

This matches the checkpoint trained in the local CXR segmentation project
(``unet_vgg11``, bilinear upsampling, 1-channel input, logits out). The
checkpoint was fit with ``BCEWithLogitsLoss``, so callers apply sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNetVGG11(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, use_batch_norm: bool = True, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        try:
            vgg11 = models.vgg11_bn(weights=None) if use_batch_norm else models.vgg11(weights=None)
        except TypeError:  # pragma: no cover - older torchvision
            vgg11 = models.vgg11_bn(pretrained=False) if use_batch_norm else models.vgg11(pretrained=False)
        features = list(vgg11.features.children())
        self.enc1 = nn.Sequential(*features[0:4])
        self.enc2 = nn.Sequential(*features[4:9])
        self.enc3 = nn.Sequential(*features[9:16])
        self.enc4 = nn.Sequential(*features[16:23])
        self.enc5 = nn.Sequential(*features[23:30])
        if n_channels == 1:
            self.enc1[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        factor = 2 if bilinear else 1
        if bilinear:
            self.dec1 = self._make_up_layer(512 + 512, 512 // factor, use_batch_norm, bilinear)
            self.dec2 = self._make_up_layer((512 // factor) + 512, 256, use_batch_norm, bilinear)
            self.dec3 = self._make_up_layer(256 + 256, 128, use_batch_norm, bilinear)
            self.dec4 = self._make_up_layer(128 + 64, 64, use_batch_norm, bilinear)
            self.dec5 = self._make_up_layer(64, 64, use_batch_norm, bilinear)
        else:
            self.dec1_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec1_conv = DoubleConv(512 + 256, 256, use_batch_norm)
            self.dec2_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2_conv = DoubleConv(256 + 128, 128, use_batch_norm)
            self.dec3_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec3_conv = DoubleConv(128 + 64, 64, use_batch_norm)
            self.dec4_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec4_conv = DoubleConv(64 + 32, 64, use_batch_norm)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def _make_up_layer(in_channels: int, out_channels: int, use_batch_norm: bool, bilinear: bool) -> nn.ModuleDict:
        up: nn.Module
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        return nn.ModuleDict({"up": up, "conv": DoubleConv(in_channels, out_channels, use_batch_norm)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        if self.bilinear:
            d1 = self.dec1["conv"](torch.cat([e4, self.dec1["up"](e5)], dim=1))
            d2 = self.dec2["conv"](torch.cat([e3, self.dec2["up"](d1)], dim=1))
            d3 = self.dec3["conv"](torch.cat([e2, self.dec3["up"](d2)], dim=1))
            d4 = self.dec4["conv"](torch.cat([e1, self.dec4["up"](d3)], dim=1))
            d5 = self.dec5["conv"](self.dec5["up"](d4))
            return self.outc(d5)
        d1 = self.dec1_conv(torch.cat([e4, self.dec1_up(e5)], dim=1))
        d2 = self.dec2_conv(torch.cat([e3, self.dec2_up(d1)], dim=1))
        d3 = self.dec3_conv(torch.cat([e2, self.dec3_up(d2)], dim=1))
        d4 = self.dec4_conv(torch.cat([e1, self.dec4_up(d3)], dim=1))
        return self.outc(d4)
