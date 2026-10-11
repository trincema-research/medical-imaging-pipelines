"""2D encoders with level-separated (x, y) heads.

Avengers (1st place): ConvNeXt-Base and EfficientNetV2-L, one 512×512 RGB
slice, five heads, L1 on (x/width, y/height).
"""

from __future__ import annotations

from rsna2024_lumbar.landmarks.constants import (
    BACKBONE_CONVNEXT_BASE,
    BACKBONE_CONVNEXT_TINY,
    BACKBONE_EFFNET_V2_L,
    BACKBONE_EFFNET_V2_S,
    LUMBAR_LEVELS,
)

_BUILDERS = {
    BACKBONE_CONVNEXT_TINY: ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    BACKBONE_CONVNEXT_BASE: ("convnext_base", "ConvNeXt_Base_Weights"),
    BACKBONE_EFFNET_V2_S: ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
    BACKBONE_EFFNET_V2_L: ("efficientnet_v2_l", "EfficientNet_V2_L_Weights"),
}


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise SystemExit("Landmark regression needs torch + torchvision. pip install -e '.[landmarks]'") from exc
    return torch, nn, models


def build_regressor(backbone: str, *, pretrained: bool = True, n_levels: int = len(LUMBAR_LEVELS)):
    """Return a 2D net: (B,3,H,W) → sigmoid xy (B,5,2) and vis logits (B,5)."""
    torch, nn, models = _require_torch()
    if backbone not in _BUILDERS:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(_BUILDERS)}.")
    fn_name, weights_name = _BUILDERS[backbone]
    kwargs = {}
    if pretrained:
        weights_enum = getattr(models, weights_name, None)
        if weights_enum is not None:
            kwargs["weights"] = weights_enum.IMAGENET1K_V1
    encoder = getattr(models, fn_name)(**kwargs)
    last = encoder.classifier[-1]
    width = int(last.in_features)
    encoder.classifier = nn.Identity()

    class LevelHeadRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.xy_heads = nn.ModuleList([nn.Linear(width, 2) for _ in range(n_levels)])
            self.vis_heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_levels)])

        def forward(self, images):
            feats = self.encoder(images)
            if feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
            xy = torch.stack([head(feats) for head in self.xy_heads], dim=1)
            vis = torch.stack([head(feats).squeeze(-1) for head in self.vis_heads], dim=1)
            return torch.sigmoid(xy), vis

    return LevelHeadRegressor()


def landmark_loss(pred_xy, pred_vis, target_xy, visible, *, vis_weight: float = 0.25):
    torch, nn, _models = _require_torch()
    mask = visible.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    loss_xy = ((pred_xy - target_xy).abs() * mask).sum() / denom
    loss_vis = nn.functional.binary_cross_entropy_with_logits(pred_vis, visible)
    return loss_xy + float(vis_weight) * loss_vis, {
        "l1": float(loss_xy.detach()),
        "bce": float(loss_vis.detach()),
    }


def pixel_errors(pred_xy, target_xy, visible, width: int, height: int) -> dict[str, float]:
    torch, _nn, _models = _require_torch()
    mask = visible > 0.5
    if int(mask.sum()) == 0:
        return {"mae_px": float("nan"), "pck5": float("nan")}
    dx = (pred_xy[..., 0] - target_xy[..., 0]) * float(width)
    dy = (pred_xy[..., 1] - target_xy[..., 1]) * float(height)
    dist = torch.sqrt(dx * dx + dy * dy)[mask]
    return {
        "mae_px": float(dist.mean().detach()),
        "pck5": float((dist <= 5.0).float().mean().detach()),
    }
